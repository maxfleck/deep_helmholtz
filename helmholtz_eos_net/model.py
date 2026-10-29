"""
EOS Model with Log Transformations (v2) - HEOSNet

Improved architecture with unified normalization for multi-task learning.
Handles both primary target (a_res) and gradient targets (from log-transformed residuals).

**Key Improvements Over v1:**
- Unified normalize_targets() for all properties
- Two-level denormalize to avoid redundancy
- Log-transformation of pressure, entropy, chemical potential residuals
- All normalization parameters stored as model attributes
- Cleaner separation between primary and gradient learning
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from helmholtz_eos_net.substance_parameters import TrainableSubstanceParameters, SubstanceParameters


def get_activation(name: Union[str, nn.Module]) -> nn.Module:
    """
    Convert activation name string to nn.Module, or return as-is if already a Module.

    Parameters:
    -----------
    name : str or nn.Module
        Activation function name ('tanh', 'relu', 'gelu', 'silu', 'leaky_relu')
        or an nn.Module instance

    Returns:
    --------
    nn.Module
        The activation function module
    """
    if isinstance(name, nn.Module):
        return name

    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "sigmoid": nn.Sigmoid(),
        "softplus": nn.Softplus(),

    }
    return activations.get(name.lower(), nn.Tanh())


class BNN(nn.Module):
    """
    Basic Neural Network with selective Dropout for regularization.

    Applies dropout only to the last hidden layer (before output) to provide
    targeted regularization for small networks without excessive capacity loss.
    """
    def __init__(self, layer_units, activation=torch.nn.Tanh(),
                 batch_norm: bool = False, dropout_rate: float = 0.0):
        """
        Creates a BNN with dropout on the last hidden layer only.

        Parameters:
        -----------
        layer_units : list[int]
            Number of nodes per layer: [input_dim, hidden_1, ..., hidden_n, output_dim]
        activation : torch.nn.Module
            Activation function (default: Tanh)
        batch_norm : bool
            If True, apply layer normalization (default: False)
        dropout_rate : float
            Dropout probability for the last hidden layer (default: 0.1)
            - 0.0: no dropout
            - 0.2-0.3: recommended for small networks

        Note: Dropout is applied ONLY to the last hidden layer (right before output).
              This is the best practice for small networks where every layer's
              capacity matters.
        """
        super(BNN, self).__init__()

        net_modules = []
        Layers = layer_units
        L = len(Layers)

        for i, (input_size, output_size) in enumerate(zip(Layers, Layers[1:])):
            layer = torch.nn.Linear(input_size, output_size)
            net_modules.append(layer)

            # Don't add activation/norm/dropout after output layer
            if i < L - 2:
                if batch_norm:
                    layer = torch.nn.LayerNorm(output_size)
                    net_modules.append(layer)

                net_modules.append(activation)

                # Add dropout ONLY to the last hidden layer (i == L - 3)
                # This means: the last iteration where i < L - 2
                if i == L - 3 and dropout_rate > 0:
                    net_modules.append(torch.nn.Dropout(dropout_rate))

        self.bnn = nn.Sequential(*net_modules)

    def forward(self, x):
        return self.bnn(x)


class GenDeepONet(torch.nn.Module):

    def __init__(self, n_branch:int=12, n_trunk:int=12,
                 d_branch:int=1, d_trunk:int=1,
                 w_branch:int=32, w_trunk:int=32,
                 p:int=24,
                 activation=torch.nn.ReLU(),
                 pass_p_branch:bool=False,
                 apply_bias:bool=False,
                 use_output_scale:bool=False,
                ):
        """
            Creates the GenDeepONet using the following parameters

            Parameters:
            n_branch (int)  : the input size of the branch network
            n_trunk  (int)  : the input size of the trunk network
            d_branch (int)  : number of layers in branch network
            d_trunk  (int)  : number of layers in trunk network
            w_branch (int)  : number of nodes at branch layer
            w_trunk  (int)  : number of nodes at trunk layer
            p        (int)  : output dimension of networks (to be mapped)
            activation      : the activation function to be used
            pass_p_branch(bool): if True branch output is returned
            apply_bias(bool): if True, apply bias to output
            use_output_scale(bool): if True, apply learnable output scaling
        """
        super(GenDeepONet, self).__init__()

        p_branch = p + n_trunk
        lu_branch = [n_branch] + d_branch*[w_branch] + [p_branch]
        lu_trunk = [n_trunk] + d_trunk*[w_trunk] + [p]

        self.p = p
        self.p_branch = p_branch
        self.n_branch = n_branch
        self.n_trunk = n_trunk
        self.pass_p_branch = pass_p_branch

        self.branch_net = BNN(lu_branch,activation=activation)
        self.trunk_net = BNN(lu_trunk,activation=activation)

        if apply_bias:
            self.bias = nn.Parameter(torch.ones((1,)),requires_grad=True)
        else:
            self.bias = None
        self.apply_bias = apply_bias

        self.use_output_scale = use_output_scale
        if use_output_scale:
            self.output_scale_param = nn.Parameter(torch.ones(1))
        else:
            self.output_scale_param = None
        return

    def forward(self,x):
        x_branch, x_trunk = torch.split(x, [self.n_branch, self.n_trunk], 1)
        x_branch = self.branch_net(x_branch)
        x_branch, xi_trunk = torch.split(x_branch, [self.p, self.n_trunk], 1)
        x_trunk = self.trunk_net(x_trunk/xi_trunk)
        y_out = torch.sum(x_branch * x_trunk, dim=1, keepdim=True)
        if self.apply_bias and self.bias is not None:
            y_out = y_out + self.bias
        if self.use_output_scale and self.output_scale_param is not None:
            y_out = y_out * self.output_scale_param
        if not self.pass_p_branch:
            return y_out
        else:
            return y_out, x_branch


class MotHONet(nn.Module):
    """
    DeepONet-style transport model for the MotH viscosity/diffusion formula.

    Two separable trunk networks matching the MotH paper's additive structure:
      de_dv = de_dv_s + de_dv_b
      de_dv_s = sum(coeff_s * trunk_s(-s_res/R / xi_s))   [free volume term]
      de_dv_b = sum(coeff_b * trunk_b(-u_res/RT / xi_b))  [energy barrier term]

    Branch outputs per molecule:
      coeff_s (p), coeff_b (p), xi_s (1), xi_b (1), neg_2_log_b, neg_log_df

    forward(x) returns (de_dv, neg_2_log_b, neg_log_df) — caller adds cc analytically.
    """
    def __init__(self, n_branch: int,
                 d_branch: int = 2, d_trunk: int = 2,
                 w_branch: int = 64, w_trunk: int = 64,
                 p: int = 8,
                 activation=nn.Tanh()):
        super().__init__()
        self.p = p
        self.n_branch = n_branch
        # branch: p coeff_s + p coeff_b + xi_s + xi_b + neg_2_log_b + neg_log_df
        p_branch_out = 2 * p + 4
        self.branch_net  = BNN([n_branch] + d_branch * [w_branch] + [p_branch_out],
                               activation=activation)
        # each trunk takes a single scalar input
        self.trunk_s_net = BNN([1] + d_trunk * [w_trunk] + [p], activation=activation)
        self.trunk_b_net = BNN([1] + d_trunk * [w_trunk] + [p], activation=activation)

    def forward(self, x):
        x_branch = x[:, :self.n_branch]
        x_s      = x[:, self.n_branch    : self.n_branch + 1]   # -s_res/R
        x_b      = x[:, self.n_branch + 1: self.n_branch + 2]   # -u_res/RT

        branch_out  = self.branch_net(x_branch)
        coeff_s     = branch_out[:, :self.p]
        coeff_b     = branch_out[:, self.p   : 2 * self.p]
        xi_s        = branch_out[:, 2*self.p     : 2*self.p + 1]
        xi_b        = branch_out[:, 2*self.p + 1 : 2*self.p + 2]
        neg_2_log_b = branch_out[:, 2*self.p + 2 : 2*self.p + 3]
        neg_log_df  = branch_out[:, 2*self.p + 3 :]

        basis_s = self.trunk_s_net(x_s / xi_s)
        basis_b = self.trunk_b_net(x_b / xi_b)
        de_dv   = (torch.sum(coeff_s * basis_s, dim=1, keepdim=True)
                 + torch.sum(coeff_b * basis_b, dim=1, keepdim=True))
        return de_dv, neg_2_log_b, neg_log_df


class MotHONet_poly(nn.Module):
    """
    MotH formula with explicit polynomial basis matching the paper:
      de_dv = poly_s(-s_res/R) + poly_b(-u_res/RT)
      poly(x) = c1*x + c2*x² + ... + cp*x^p  (no constant term)
    Branch maps SAFT params → (coeff_s[p], coeff_b[p], neg_2_log_b, neg_log_df).
    p controls polynomial degree (default 3 = cubic, matching MotH paper).

    forward(x) returns (de_dv, neg_2_log_b, neg_log_df) — caller adds cc analytically.
    """
    def __init__(self, n_branch: int,
                 d_branch: int = 2, w_branch: int = 64,
                 p: int = 3,
                 activation=nn.Tanh(),
                 # kept for API compat, unused:
                 d_trunk: int = 2, w_trunk: int = 64):
        super().__init__()
        self.n_branch = n_branch
        self.p = p
        # Typical scales for liquid water: -s_res/R ≈ 6, -u_res/(RT) ≈ 16.
        # Dividing inputs by these scales keeps all polynomial powers O(1), preventing
        # gradient explosion through x_b^p with large x_b (e.g. 16^3 = 4096).
        self.x_s_ref = 6.0
        self.x_b_ref = 16.0
        # branch: p coeff_s + p coeff_b + neg_2_log_b + neg_log_df
        self.branch_net = BNN([n_branch] + d_branch * [w_branch] + [2 * p + 2],
                              activation=activation)
        # Zero-init ONLY polynomial coefficient rows to prevent overflow.
        # neg_2_log_b and neg_log_df rows keep Kaiming init so hidden layers get non-zero gradients.
        with torch.no_grad():
            self.branch_net.bnn[-1].weight[:2 * p].zero_()
            self.branch_net.bnn[-1].bias[:2 * p].zero_()

    def forward(self, x):
        x_branch = x[:, :self.n_branch]
        x_s      = x[:, self.n_branch    : self.n_branch + 1]   # -s_res/R
        x_b      = x[:, self.n_branch + 1: self.n_branch + 2]   # -u_res/RT

        branch_out  = self.branch_net(x_branch)
        coeff_s     = branch_out[:, :self.p]
        coeff_b     = branch_out[:, self.p : 2 * self.p]
        neg_2_log_b = branch_out[:, 2 * self.p     : 2 * self.p + 1]
        neg_log_df  = branch_out[:, 2 * self.p + 1 :]

        # Normalize polynomial inputs to O(1) range before computing powers
        z_s = x_s / self.x_s_ref
        z_b = x_b / self.x_b_ref

        # polynomial basis [z, z², ..., z^p], no constant term
        powers_s = torch.cat([z_s ** i for i in range(1, self.p + 1)], dim=1)
        powers_b = torch.cat([z_b ** i for i in range(1, self.p + 1)], dim=1)
        #print(powers_s.shape, coeff_s.shape)
        #print(powers_b.shape, coeff_b.shape)

        de_dv = (torch.sum(coeff_s * powers_s, dim=1, keepdim=True)
               + torch.sum(coeff_b * powers_b, dim=1, keepdim=True))
        #print(de_dv.shape, neg_2_log_b.shape, neg_log_df.shape)
        return de_dv, neg_2_log_b, neg_log_df


class HEOSNet(torch.nn.Module):
    """
    Helmholtz Energy EOS with Log Transformations (v2).

    Unified normalization/denormalization for primary target (a_res)
    and gradient targets (extracted from log-transformed s_res, p_res, mu_res).

    **Architecture:**
    - Unified normalize_targets() handles all targets
    - Two-level denormalize: physical properties → log-residuals (inner)

    **Normalization:**
    - All min/max bounds stored as model parameters
    - Computed from synthetic training data during initialization
    - Log-transformed residuals kept in raw log-space (not rescaled to [0,1])
    """

    def __init__(self,
                 substance_params: Optional[Union['TrainableSubstanceParameters', 'SubstanceParameters']] = None,
                 transport_params: Optional[Union['TrainableSubstanceParameters', 'SubstanceParameters']] = None,
                 p_dim: int = None,
                 depth: int = 4,
                 width: int = 48,
                 activation: Union[str, nn.Module] = "gelu",
                 dropout_rate: float = 0.0,
                 transport_net_hidden: int = 0,
                 transport_net_layers: int = 2,
                 transport_net_p: int = 8):
        """
        Initialize HEOSNet.

        Parameters:
        -----------
        substance_params : SubstanceParameters
            Substance parameter manager
        p_dim : int
            Substance embedding dimension (auto-detected from substance_params)
        depth : int
            Hidden layers in the Helmholtz network (default: 4)
        width : int
            Neurons per hidden layer (default: 48)
        activation : str or nn.Module
            Activation function (default: 'gelu')
        dropout_rate : float
            Dropout on the last hidden layer (default: 0.0)
        transport_net_hidden : int
            Hidden width of MotHONet branch/trunk; 0 = no transport net (default: 0)
        transport_net_layers : int
            Number of hidden layers in MotHONet branch/trunk (default: 2)
        transport_net_p : int
            Number of DeepONet modes (basis functions) in MotHONet (default: 8)
        """
        super(HEOSNet, self).__init__()

        # Substance parameters (EOS)
        if substance_params is not None:
            self.substance_params = substance_params
            if p_dim is None:
                p_dim = substance_params.embedding_dim
        else:
            self.substance_params = None

        self.p_dim = p_dim

        # Convert activation string to module if needed
        activation_module = get_activation(activation)

        # Helmholtz network
        print(f"Network config: p_dim={p_dim}, depth={depth}, width={width}, dropout={dropout_rate}")
        arch = [p_dim + 2] + depth * [width] + [1]
        self.simple_net = BNN(arch, activation=activation_module,
                              batch_norm=False, dropout_rate=dropout_rate)
        self.activation = activation_module
        self.activation_name = activation if isinstance(activation, str) else type(activation_module).__name__.lower()
        print(f"Architecture: {arch}")

        # MotHONet — built here if transport_net_hidden > 0, else set externally
        # Switch between MotHONet (BNN trunks) and MotHONet_poly (polynomial basis) here.
        _MotHONetCls = MotHONet_poly
        if transport_params is not None:
            self.transport_params = transport_params
        elif transport_net_hidden > 0:
            self.transport_params = _MotHONetCls(
                n_branch=p_dim,
                d_branch=transport_net_layers,
                d_trunk=transport_net_layers,
                w_branch=transport_net_hidden,
                w_trunk=transport_net_hidden,
                p=transport_net_p,
                activation=activation_module,
            )
            n_tp = sum(p.numel() for p in self.transport_params.parameters())
            print(f"{_MotHONetCls.__name__}: n_branch={p_dim}, p={transport_net_p}, "
                  f"hidden={transport_net_hidden}×{transport_net_layers}, params={n_tp}")
        else:
            self.transport_params = None

        self._rho_cache: Optional[torch.Tensor] = None  # shape (N, 4): [sub_id, T, P, rho]
        self.transport_rho_n_steps: int = 20  # Newton steps for density search in transport forward
        

        # Input normalization (log-transformed space)
        self.register_buffer('min_T_log', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('max_T_log', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('T_log_range', torch.tensor(1.0, dtype=torch.float32))

        self.register_buffer('min_rho_log', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('max_rho_log', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('rho_log_range', torch.tensor(1.0, dtype=torch.float32))

        # Primary output normalization (log(-a_res))
        self.register_buffer('min_a_res_log', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('max_a_res_log', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('a_res_log_range', torch.tensor(1.0, dtype=torch.float32))

        # Label rescaling for numerical stability (computed during set_normalization_from_data)
        self.register_buffer('label_scale', torch.tensor(1.0, dtype=torch.float32))        


    def forward_normalized(self, p_input, x_in_norm, 
                           compute_gradients=True,
                           compute_gradient2_rhorho=False,
                           compute_transport=False,
                           ):
        """
        Core forward pass: predict normalized a_res and optionally gradients via autograd.

        Takes pre-computed molecule embeddings and normalized state point inputs.
        Computes gradients using torch.autograd.grad() if compute_gradients=True.

        Args:
            p_input: molecule embeddings, shape (batch_size, p_dim)
            x_in_norm: Normalized [T, rho], shape (batch_size, 2)
            compute_gradients: If True, compute and return gradients; if False, return only a_res

        Returns (when compute_gradients=True):
            Tensor: Concatenated predictions, shape (batch_size, 3)
            - Column 0: Normalized a_res
            - Column 1: Gradient ∂(log a)/∂T_norm
            - Column 2: Gradient ∂(log a)/∂(log ρ)_norm

        Returns (when compute_gradients=False):
            Single tensor: y_pred_norm, shape (batch_size, 1) - a_res only
        """
        # Enable gradient tracking if computing gradients
        if compute_gradients:
            x_in_norm = x_in_norm.requires_grad_(True)

        # HEOSNet expects concatenated input: [p_input, x_in_norm]
        x_combined = torch.cat([p_input, x_in_norm], dim=1)

        # Forward through HEOSNet
        y_pred_norm = self.simple_net(x_combined)

        # Ideal gas limit: rho_norm = log(rho), so rho_norm <= 3 means rho <= e³ ≈ 20 mol/m³.
        # Zeroing y_pred_norm here blocks autograd → grad_T = grad_rho = 0 for those rows,
        # so p_res = s_res = mu_res = 0 exactly; a_res = -exp(0) = -1 J/mol (≈0, negligible).
        # Threshold 20 mol/m³ covers ideal gas at T up to ~400 K (p_sat up to ~50 kPa).
        # ideal_gas_mask = (x_in_norm[:, 1:2] <= 1)
        #ideal_gas_mask = (x_in_norm[:, 1:2] <= -3)
        ideal_gas_mask = (x_in_norm[:, 1:2] <= -30)
        if ideal_gas_mask.any():
            y_pred_norm = torch.where(ideal_gas_mask, torch.zeros_like(y_pred_norm), y_pred_norm)

        if compute_gradient2_rhorho:
            compute_gradients = True

        # Compute gradients via autograd
        if compute_gradients:
            grads_normalized = torch.autograd.grad(
                outputs=y_pred_norm.sum(),
                inputs=x_in_norm,
                create_graph=True,
                retain_graph=True
            )[0]
            grad_T_pred_norm = grads_normalized[:, 0:1]
            grad_rho_pred_norm = grads_normalized[:, 1:2]
            # Concatenate into single tensor: (batch, 3)

            if compute_gradient2_rhorho:
                # ∂²a/∂ρ²
                grad2_rhorho = torch.autograd.grad(
                    outputs=grad_rho_pred_norm.sum(),
                    inputs=x_in_norm,
                    create_graph=True,
                    retain_graph=True
                )[0][:, 1:2]
                return torch.cat([y_pred_norm, grad_T_pred_norm, grad_rho_pred_norm, grad2_rhorho], dim=1)  
            else:
                return torch.cat([y_pred_norm, grad_T_pred_norm, grad_rho_pred_norm], dim=1)
        else:
            # Only return a_res when gradients not computed (Phase 1 training)
            return y_pred_norm

    def forward_train(self, substance_ids, x_in, compute_gradients=True, compute_gradient2_rhorho=False):
        """
        Forward pass for training: returns normalized predictions with optional gradients.

        Automatically normalizes inputs and looks up embeddings by ID.
        Computes gradients via autograd for training loss.

        Args:
            substance_ids: Substance indices, shape (batch_size,)
            x_in: input in physical units [T, rho], shape (batch_size, 2)
            compute_gradients: If True, compute gradients via autograd (default: True)

        Returns (when compute_gradients=False):
            Single tensor: y_pred_norm, shape (batch_size, 1) - normalized a_res only for Phase 1 training

        Returns (when compute_gradients=True):
            Tensor: Concatenated normalized properties, shape (batch_size, 4)
            - Column 0: Normalized a_res
            - Column 1: Normalized s_res (computed from ∂a/∂T)
            - Column 2: Normalized pressure (computed from ∂a/∂ρ + ideal gas)
            - Column 3: Normalized mu_res (computed from a_res and ∂a/∂ρ)
        """
        if self.substance_params is None:
            raise RuntimeError("forward_train() requires substance_params to be set")

        p_input = self.substance_params(substance_ids)

        # Normalize t, rho inputs
        x_in_norm = self.normalize_input(x_in)

        if not compute_gradients:
            # Phase 1: Return normalized a_res only
            return self.forward_normalized(p_input, x_in_norm, compute_gradients=False)
        # Phase 2: Return predicted properties based on training_mode
        result = self.forward_normalized(p_input, x_in_norm, compute_gradients=True, 
                                        compute_gradient2_rhorho=compute_gradient2_rhorho)
        y_norm = result[:, 0:1]           # Normalized a_res
        grad_T = result[:, 1:2]           # Raw ∂a/∂T
        grad_rho = result[:, 2:3]         # Raw ∂a/∂ρ

        if compute_gradient2_rhorho:
            grad2_rhorho = result[:, 3:4]  # Raw ∂²a/∂ρ²
            return torch.cat([y_norm, grad_T, grad_rho, grad_rho, grad2_rhorho], dim=1)
        else:
            # Return 4 normalized values: (a_res, grad_T, grad_rho_p, grad_rho_mu)
            return torch.cat([y_norm, grad_T, grad_rho, grad_rho], dim=1)

    def forward(self, p_input, x_in, compute_gradient2_rhorho=False):
        """
        Forward pass for inference: returns physical properties (denormalized).

        User-facing method that automatically normalizes inputs and returns
        physically meaningful property values. Always computes gradients for
        full property inference.

        Args:
            p_input: parameter embeddings, shape (batch_size, p_dim)
            x_in: input in physical units [T, rho], shape (batch_size, 2)

        Returns (all RESIDUAL quantities — ideal gas NOT included):
            a_res:    Residual Helmholtz energy (J/mol),    shape (batch_size, 1)
            s_res:    Residual entropy (J/mol/K),           shape (batch_size, 1)
            p_res:    Residual pressure (Pa),               shape (batch_size, 1)
            mu_res:   Residual chemical potential (J/mol),  shape (batch_size, 1)
            [dp_drho: Residual dp/drho (Pa·m³/mol), only if compute_gradient2_rhorho=True]
            [dmu_drho: Residual dmu/drho, only if compute_gradient2_rhorho=True]

        Total pressure:  p_total  = p_res  + R * T * rho
        Total dp/drho:   dp_total = dp_drho + R * T
        """
        # Normalize T,p inputs
        x_in_norm = self.normalize_input(x_in)

        # Get normalized predictions with gradients
        result = self.forward_normalized(p_input, x_in_norm, compute_gradients=True,
                                        compute_gradient2_rhorho=compute_gradient2_rhorho)

        # Unpack concatenated tensor
        y_pred_norm = result[:, 0:1]
        grad_T_pred_norm = result[:, 1:2]
        grad_rho_pred_norm = result[:, 2:3]

        if compute_gradient2_rhorho:
            grad_rho_rho_pred_norm = result[:, 3:4]
            # Denormalize to physical units (returns 6 values)
            return self.residuals_from_gradients(
                y_pred_norm, grad_T_pred_norm, grad_rho_pred_norm, x_in_norm, grad_rho_rho_pred_norm
            )
        else:
            # Denormalize to physical units (returns 4 values)
            return self.residuals_from_gradients(
                y_pred_norm, grad_T_pred_norm, grad_rho_pred_norm, x_in_norm
            )

    def extract_hidden_space_features(self, p_input, x_in, layer: int = -1,
                                      normalized_input: bool = False) -> torch.Tensor:
        """
        Return the hidden-layer activations of the Helmholtz network.

        Args:
            p_input: substance embeddings, shape (batch_size, p_dim)
            x_in: [T, rho] in physical units (K, mol/m³), shape (batch_size, 2).
                  If normalized_input=True, x_in is taken as already normalized.
            layer: hidden layer index (0 ... depth-1); negative indices count
                   from the end. Default -1 = last hidden layer.
            normalized_input: skip input normalization if True.

        Returns:
            Post-activation features of the requested hidden layer,
            shape (batch_size, width). The output Linear layer is never applied.
        """
        modules = list(self.simple_net.bnn)
        n_hidden = sum(1 for m in modules if isinstance(m, nn.Linear)) - 1
        if not -n_hidden <= layer < n_hidden:
            raise IndexError(f"layer={layer} out of range for {n_hidden} hidden layers "
                             f"(valid: {-n_hidden} ... {n_hidden - 1})")
        target = layer % n_hidden

        x_in_norm = x_in if normalized_input else self.normalize_input(x_in)
        x = torch.cat([p_input, x_in_norm], dim=1)

        hidden_idx = -1
        for i, module in enumerate(modules):
            x = module(x)
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Dropout)):
                continue
            # activation module -> end of a hidden layer (include trailing dropout)
            hidden_idx += 1
            if hidden_idx == target:
                if i + 1 < len(modules) and isinstance(modules[i + 1], nn.Dropout):
                    x = modules[i + 1](x)
                return x
        raise RuntimeError("Requested hidden layer not reached")  # should not happen


    def normalize_input(self, x_in):
        """
        Normalize physical inputs: log-transform then normalize to [0, 1].

        Args:
            x_in: [T, rho], shape (batch_size, 2)

        Returns:
            x_in_norm: Normalized [log(T), log(rho)], shape (batch_size, 2)
        """
        T_norm = self._normalize_T(x_in[:, 0:1])
        rho_norm = self._normalize_rho(x_in[:, 1:2])

        return torch.cat([T_norm, rho_norm], dim=1)

    # ========== UNIFIED NORMALIZATION (Primary Entry Point) ==========
    def normalize_targets(self, y_everything, x_in, y_pred_norm=None):
        """
        Normalize targets: a_res (primary) and either gradients or residuals based on training_mode.

        Uses EXPERIMENTAL a_res to compute target gradients, avoiding numerical instability
        from divisions by predicted a_res.

        Applies label rescaling before normalization for improved numerical stability.

        Args:
            y_everything: [a_res, s_res, pressure, mu_res], shape (batch_size, 4)
            x_in: [T, rho], shape (batch_size, 2) in physical space
            y_pred_norm: model predictions [a_res_norm, grad_T, grad_rho_p, grad_rho_mu],
                shape (batch_size, 4). When provided, also returns ln_a_res from each residual.
        """

        a_res = y_everything[:, 0:1]
        s_res = y_everything[:, 1:2]
        p_res = y_everything[:, 2:3]
        mu_res = y_everything[:, 3:4]

        # 1. Normalize primary target
        a_res_norm = self._normalize_a_res(a_res)

        # 2. Compute secondary targets based on training mode
        grad_T = self._extract_grad_T_from_s_res(s_res, x_in[:, 0:1], a_res)
        grad_rho_p = self._extract_grad_rho_from_p_res(p_res, x_in[:, 1:2], a_res)
        grad_rho_mu = self._extract_grad_rho_from_mu_res(mu_res, a_res)

        # 3. Compute a_res from gradients (when model predictions are provided)
        if y_pred_norm is not None:
            grad_T = y_pred_norm[:, 1:2]
            grad_rho = y_pred_norm[:, 2:3]
            a_res_s = self._extract_a_res_from_s_res(s_res, x_in[:, 0:1], grad_T)
            ln_a_res_s = self._normalize_a_res(a_res_s)
            a_res_p = self._extract_a_res_from_p_res(p_res, x_in[:, 1:2], grad_rho)
            ln_a_res_p = self._normalize_a_res(a_res_p)
            a_res_mu = self._extract_a_res_from_mu_res(mu_res, grad_rho)
            ln_a_res_mu = self._normalize_a_res(a_res_mu)
            return torch.cat([a_res_norm, grad_T, grad_rho_p, grad_rho_mu,
                              ln_a_res_s, ln_a_res_p, ln_a_res_mu], dim=1)
        else:
            return torch.cat([a_res_norm, grad_T, grad_rho_p, grad_rho_mu], dim=1)

    # ========== TWO-LEVEL DENORMALIZATION ==========
    def residuals_from_gradients(self, y_norm, grad_T_norm, grad_rho_norm,
                                         x_in_norm, grad_rho_rho_norm: Optional[torch.Tensor]=None):
        """
        Level 1 (Outer): Compute physical properties from normalized predictions.

        Calls inner denormalization to avoid redundancy.

        Args:
            y_norm: Normalized a_res, shape (batch_size, 1)
            grad_T_norm: Normalized temperature gradient, shape (batch_size, 1)
            grad_rho_norm: Normalized density gradient, shape (batch_size, 1)
            x_in_norm: Normalized [T, rho], shape (batch_size, 2)
            grad_rho_rho_norm: Normalized 2nd density gradient, shape (batch_size, 1)

        Returns:
            a_res:    Residual Helmholtz energy (J/mol), shape (batch_size, 1)
            s_res:    Residual entropy (J/mol/K),        shape (batch_size, 1)
            p_res:    Residual pressure (Pa),            shape (batch_size, 1)  ← NOT total pressure
            mu_res:   Residual chemical potential (J/mol), shape (batch_size, 1)
            dp_drho:  Residual dp/drho (Pa·m³/mol),     shape (batch_size, 1)  [only if compute_gradient2_rhorho]
            dmu_drho: Residual dmu/drho (J·m³/mol²),    shape (batch_size, 1)  [only if compute_gradient2_rhorho]

        To get total pressure: p_total = p_res + R * T * rho
        To get total dp/drho:  dp_total = dp_drho + R * T
        """
        # Denormalize inputs
        T = self._denormalize_T(x_in_norm[:, 0:1])
        rho = self._denormalize_rho(x_in_norm[:, 1:2])

        # Denormalize primary output
        a_res = self._denormalize_a_res(y_norm)

        # Level 2 (Inner): Convert normalized gradients to log-residuals
        s_res = self._invert_grad_T(grad_T_norm, T, a_res)
        p_res = self._invert_grad_rho_p(grad_rho_norm, rho, a_res)
        mu_res = self._invert_grad_rho_mu(grad_rho_norm, a_res)

        if grad_rho_rho_norm is not None:
            dp_drho = self._invert_grad_rho_rho_p(grad_rho_norm, grad_rho_rho_norm, a_res)
            dmu_drho = self._invert_grad_rho_rho_mu(grad_rho_norm, grad_rho_rho_norm, rho, a_res)
            return a_res, s_res, p_res, mu_res, dp_drho, dmu_drho

        return a_res, s_res, p_res, mu_res

    def find_vle_flash_batch(self, p_input, T_batch, rho_gas_init, rho_liquid_init,
                            n_steps=10, tol=1e-5, alpha=0.5, gas_reset_factor=0.95,
                            liq_reset_factor=1.05, via_gradient2_rhorho=True, h_rel=0.001,
                            convergence_threshold=1000.0, determine_rho_liq=None):
        """
        Batch VLE flash via Maxwell construction + independent scalar Newton per phase.

        Replaces the ill-conditioned 2x2 Jacobian of xfind_vle_flash_batch with a
        decoupled approach inspired by feos (iterate_pure_t in vle_pure.rs):

          Step 1 — Maxwell pressure estimate from current densities:
            delta_a = a_res_gas - a_res_liq + RT*ln(rho_gas/rho_liq)
            delta_v = 1/rho_gas - 1/rho_liq
            p_sat   = -delta_a / delta_v          (thermodynamically exact given current rhos)

          Step 2 — Negative-pressure fallback (low T: p_sat can underflow):
            p_sat = p_gas * exp((-delta_a - p_gas/rho_gas) / RT)   (ideal-vapor limit)

          Step 3 — Independent scalar Newton per phase:
            drho_gas = (p_sat - p_gas) / dp_gas
            drho_liq = (p_sat - p_liq) / dp_liq

        Stability benefit: at low T, dmu/drho_gas = RT/rho_gas → ∞. The old 2x2 determinant
        det = dmu_gas*dp_liq - dmu_liq*dp_gas has catastrophic cancellation → large noisy steps.
        The Maxwell formula uses only a_res and p, which are always well-conditioned.

        See xfind_vle_flash_batch for the old 2x2 approach (kept for reference).

        Parameters of note
        ------------------
        tol : float
            Relative Newton step |alpha*drho|/rho below which a phase counts as converged.
            Default 1e-5: with alpha=0.5 damping and the float32 liquid density noise floor
            of drho/rho ~ 8e-7, the step cannot reliably go below ~1e-6. Can be tightened
            toward 1e-7 in float64.
        convergence_threshold : float
            IGNORED — accepted for backward compatibility with existing call sites only.
            The old |p_gas - p_liq| < 1000 Pa test was ill-conditioned on the liquid branch
            and vacuously true below ~285 K where p_sat < 1000 Pa. Use `tol` instead.
        """
        R = 8.314
        n_temps = len(T_batch)

        T_merged = torch.cat([T_batch, T_batch])  # (2*n_temps, 1)
        rho = torch.cat([
            rho_gas_init.detach().clone(),
            rho_liquid_init.detach().clone(),
        ])

        p_merged = torch.cat([p_input, p_input], dim=0)  # (2*n_temps, p_dim)

        if not via_gradient2_rhorho:
            T_stacked = torch.cat([T_merged, T_merged])   # (4*n_temps, 1)
            p_stacked = torch.cat([p_merged, p_merged], dim=0)

        # Initial p_sat estimate for feos inner loop (ideal-gas approximation)
        p_sat_prev = (R * T_batch * rho_gas_init.detach().clamp(min=1e-6)).clamp(min=1e-10)

        for step in range(n_steps):
            if step > 0:
                rho = rho.detach()

            if via_gradient2_rhorho:
                x_in = torch.cat([T_merged, rho], dim=1)
                a_res, _, p_res, _, dp_drho, _ = self.forward(
                    p_merged, x_in, compute_gradient2_rhorho=True
                )
                p = p_res + R * T_merged * rho
                dp_drho = dp_drho + R * T_merged
            else:
                rho_stacked = torch.cat([rho, rho * (1 + h_rel)])
                h = rho * h_rel
                x_in = torch.cat([T_stacked, rho_stacked], dim=1)
                a_res_all, _, p_res_all, _ = self.forward(p_stacked, x_in)
                p_all = p_res_all + R * T_stacked * rho_stacked
                a_res = a_res_all[:2 * n_temps]
                p = p_all[:2 * n_temps]
                p_plus = p_all[2 * n_temps:]
                dp_drho = (p_plus - p) / h

            violated = dp_drho <= 0

            rho_gas = rho[:n_temps]
            rho_liq = rho[n_temps:]
            a_res_gas = a_res[:n_temps]
            a_res_liq = a_res[n_temps:]
            p_gas = p[:n_temps]
            p_liq = p[n_temps:]
            dp_gas = dp_drho[:n_temps]
            dp_liq = dp_drho[n_temps:]
            violated_gas = violated[:n_temps]
            violated_liq = violated[n_temps:]

            # ── TEST (temporary, hard-coded for water): ideal vapour below T_IG ──
            # Comment this block out to restore the NN gas phase.
            #T_IG  = 320.0
            #ig    = T_batch < T_IG          # (n_temps, 1) bool
            #RT_ig = R * T_batch             # (n_temps, 1)
            #a_res_gas    = torch.where(ig, torch.zeros_like(a_res_gas), a_res_gas)
            #p_gas        = torch.where(ig, RT_ig * rho_gas, p_gas)
            #dp_gas       = torch.where(ig, RT_ig, dp_gas)
            #violated_gas = violated_gas & ~ig
            # ── end TEST ──

            # Maxwell construction: p_sat from Helmholtz energy balance
            v_gas = 1.0 / rho_gas.clamp(min=1e-6)
            v_liq = 1.0 / rho_liq.clamp(min=1e-6)
            delta_v = v_gas - v_liq
            delta_a = (a_res_gas - a_res_liq
                       + R * T_batch * (rho_gas / rho_liq.clamp(min=1e-6)).log())
            safe_dv = torch.where(delta_v.abs() < 1e-20,
                                  torch.ones_like(delta_v) * 1e-20, delta_v)
            p_sat_est = -delta_a / safe_dv

            # Negative-pressure fallback: ideal-gas approximation (low-T regime)
            RT = R * T_batch
            p_sat_est = torch.where(
                p_sat_est < 0,
                # p_gas * torch.exp((-delta_a - p_gas * v_gas) / RT.clamp(min=1e-10)),
                rho_gas * RT,
                p_sat_est,
            ).clamp(min=1e-10)

            # Inner Newton loop: refine p_sat_est using ideal-gas correction (feos approach).
            # Solves f(p) = p·Δv + Δa + RT·(ln(p/p_prev) + 1 − p/p_prev) = 0 analytically,
            # accounting for the change in ideal-gas vapor density as p shifts.
            # No extra forward passes — uses delta_a and delta_v already in hand.
            p_prev = p_sat_prev.clamp(min=1e-10)
            for _ in range(20):
                p_frac = p_sat_est / p_prev
                f_inner = (p_sat_est * delta_v + delta_a
                           + RT * (p_frac.clamp(min=1e-30).log() + 1.0 - p_frac))
                df_inner = delta_v + RT * (1.0 / p_sat_est.clamp(min=1e-10) - 1.0 / p_prev)
                safe_df = torch.where(df_inner.abs() < 1e-30,
                                      torch.ones_like(df_inner) * 1e-30, df_inner)
                p_sat_est = (p_sat_est - f_inner / safe_df).clamp(min=1e-10)
            p_sat_prev = p_sat_est.detach()

            # Independent scalar Newton per phase
            drho_gas    = (p_sat_est - p_gas) / dp_gas.clamp(min=1.0)
            drho_liquid = (p_sat_est - p_liq) / dp_liq.clamp(min=1.0)

            if determine_rho_liq is not None:
                drho_liquid = drho_liquid * determine_rho_liq

            rho_gas_new = torch.where(~violated_gas, rho_gas + alpha * drho_gas, rho_gas)
            rho_liq_new = torch.where(~violated_liq, rho_liq + alpha * drho_liquid, rho_liq)

            rho_new = torch.clamp(torch.cat([rho_gas_new, rho_liq_new]), min=1e-6, max=100000)
            rho = torch.where(torch.isfinite(rho_new), rho_new, rho)

        rho = rho.detach()
        rho_gas_converged = rho[:n_temps, 0:1]
        rho_liq_converged = rho[n_temps:, 0:1]

        # p_sat = p_res + RT·rho is exact on either branch; only the conditioning differs.
        # Liquid: RT·rho_liq / p_sat ~ 1e4–1e5, so float32 cancellation destroys p_sat at low T.
        # Vapour: RT·rho_gas / p_sat = 1/Z_gas, bounded by ~1.8 up to the critical point.
        # So evaluate it on the vapour branch. Dropping p_res_gas (pure ideal gas) is only valid
        # at low T — it costs 36% MAPE above 500 K where Z_gas falls to 0.56.
        x_final = torch.cat([
            torch.cat([T_batch, rho_gas_converged], dim=1),
            torch.cat([T_batch, rho_liq_converged], dim=1),
        ], dim=0)
        _, _, p_res_both, mu_res_both = self.forward(torch.cat([p_input, p_input], dim=0), x_final)
        p_res_gas_final  = p_res_both[:n_temps]      # vapour branch -> p_sat
        mu_res_liq_final = mu_res_both[n_temps:]     # liquid branch -> mu_sat

        p_sat_pred  = p_res_gas_final + R * T_batch * rho_gas_converged
        mu_sat_pred = mu_res_liq_final + R * T_batch * torch.log(rho_liq_converged.clamp(min=1e-6))
        p_sat_pred = torch.nan_to_num(p_sat_pred, nan=1.0, posinf=1e6, neginf=-1e6)
        mu_sat_pred = torch.nan_to_num(mu_sat_pred, nan=1.0, posinf=1e6, neginf=-1e6)

        # Convergence on the density iterate, not on |p_gas - p_liq|. That difference is
        # unusable here: p_liq is a 1e4-1e5x cancelling sum (see the note above), the absolute
        # 1000 Pa threshold exceeds p_sat itself below ~285 K, and p_gas/p_liq are the values
        # from *before* the last density update. The relative Newton step is conditioning-
        # immune: the same dp/drho that destroys p_liq divides the error back out, leaving
        # drho/rho ~ 1e-6. drho_* / rho_* / violated_* are the last iteration's values, so
        # this is the standard post-hoc Newton test on the step just taken.
        rel_step_gas = (drho_gas * alpha).abs() / rho_gas.clamp(min=1e-6)
        rel_step_liq = (drho_liquid * alpha).abs() / rho_liq.clamp(min=1e-6)
        converged_mask = (
            (rel_step_gas < tol) & (rel_step_liq < tol)
            & ~violated_gas & ~violated_liq
            & torch.isfinite(rho_gas_converged) & torch.isfinite(rho_liq_converged)
        )

        return p_sat_pred, mu_sat_pred, rho_gas_converged, rho_liq_converged, converged_mask


    def find_rho_batch(self, p_input, T_batch, P_target, rho_init=None,
                       n_steps=20, alpha=0.5, h_rel=0.001,
                       via_gradient2_rhorho=True,
                       convergence_threshold=100.0):
        """
        Batch density iteration: find rho given T and P_target via scalar Newton.

        Solves f(rho) = p(T, rho) - P_target = 0 using dp/drho from the second
        derivative (autograd) or finite differences.

        Parameters
        ----------
        p_input : Tensor (batch, p_dim)
            Substance embedding, from self.substance_params(substance_ids).
        T_batch : Tensor (batch, 1)
            Temperature in K (physical units).
        P_target : Tensor (batch, 1)
            Target pressure in Pa (physical units).
        rho_init : Tensor (batch, 1) or None
            Initial density guess in mol/m³. If None, uses ideal gas: P/(R*T).
        n_steps : int
            Newton iterations (default 20).
        alpha : float
            Step damping factor (default 0.5).
        h_rel : float
            Relative FD step size (only used when via_gradient2_rhorho=False).
        via_gradient2_rhorho : bool
            If True, use autograd second derivative for dp/drho (default).
            If False, use finite differences.
        convergence_threshold : float
            |p_pred - P_target| < threshold to mark converged (Pa, default 100).

        Returns
        -------
        p_pred : Tensor (batch, 1) — final predicted pressure (Pa)
        rho_converged : Tensor (batch, 1) — converged density (mol/m³)
        converged_mask : Tensor (batch, 1) — bool, True if converged
        """
        R = 8.314
        max_rho = 100000.0

        if rho_init is None:
            rho = (P_target / (R * T_batch)).clamp(min=1e-6, max=max_rho)
        else:
            rho = rho_init.detach().clone().clamp(min=1e-6, max=max_rho)

        if not via_gradient2_rhorho:
            T_stacked = torch.cat([T_batch, T_batch])
            p_stacked = torch.cat([p_input, p_input], dim=0)

        n = len(T_batch)

        for step in range(n_steps):
            if step > 0:
                rho = rho.detach()

            if via_gradient2_rhorho:
                x_in = torch.cat([T_batch, rho], dim=1)
                _, _, p_res, _, dp_drho_res, _ = self.forward(
                    p_input, x_in, compute_gradient2_rhorho=True
                )
                p = p_res + R * T_batch * rho
                dp_drho = dp_drho_res + R * T_batch
            else:
                rho_stacked = torch.cat([rho, rho * (1 + h_rel)])
                h = rho * h_rel
                x_in = torch.cat([T_stacked, rho_stacked], dim=1)
                _, _, p_res_all, _ = self.forward(p_stacked, x_in)
                p_all = p_res_all + R * T_stacked * rho_stacked
                p = p_all[:n]
                p_plus = p_all[n:]
                dp_drho = (p_plus - p) / h

            spinodal = dp_drho <= 0
            f = p - P_target
            drho = -f / dp_drho.clamp(min=1e-30)
            # feos step limiter: cap at ±7.5% of max_rho, prevent negative density
            step_limit = 0.075 * max_rho
            drho = drho.clamp(min=-step_limit, max=step_limit)
            drho = torch.max(drho, -0.95 * rho)
            # Freeze in spinodal (NN artifact — don't jump branches)
            drho = torch.where(spinodal, torch.zeros_like(drho), drho)

            rho_new = (rho + alpha * drho).clamp(min=1e-6, max=max_rho)
            rho = torch.where(torch.isfinite(rho_new), rho_new, rho)

        rho = rho.detach()

        x_final = torch.cat([T_batch, rho], dim=1)
        _, _, p_res_final, _ = self.forward(p_input, x_final)
        p_pred = p_res_final + R * T_batch * rho
        p_pred = torch.nan_to_num(p_pred, nan=1.0, posinf=1e9, neginf=-1e9)

        # feos-style hybrid convergence: absolute floor of 1 Pa, relative 1e-6 above that.
        converged_mask = (p_pred - P_target).abs() < torch.clamp(rho * 1e-6, min=1.0)
        return p_pred, rho, converged_mask

    def find_rho_norm(self, substance_ids, T_batch, P_target, rho_init=None, **kwargs):
        """find_rho_batch with substance IDs instead of pre-computed p_input."""
        p_input = self.substance_params(substance_ids)
        return self.find_rho_batch(p_input, T_batch, P_target, rho_init, **kwargs)

    # =========================================================================
    # TRANSPORT PROPERTIES (MotH model)
    # Requires self.transport_params to be set (4 dims: b, n, m, df).
    # =========================================================================

    def forward_transport_train(self, substance_ids, T, P, mw, rho_init=None):
        """
        Core transport forward. Takes physical inputs, returns training outputs.

        Returns
        -------
        result     : Tensor (batch, 5) — [a_res_norm, grad_T_norm, grad_rho_norm, ln_vis, ln_diff]
        x_in       : Tensor (batch, 2) — [T_phys, rho_phys]
        a_res      : Tensor (batch, 1) — physical residual Helmholtz [J/mol]
        None, None : placeholders (indirect goal inversion removed)
        """
        R = 8.31446261815324

        p_input = self.substance_params(substance_ids)
        if self._helmholtz_is_frozen:
            rho = self._rho_cache_lookup(substance_ids, T, P)
            if rho is None:
                _, rho, _ = self.find_rho_batch(p_input, T, P, rho_init=rho_init,
                                                n_steps=self.transport_rho_n_steps)
                rho = rho.detach()
                self._rho_cache_append(substance_ids, T, P, rho)
        else:
            _, rho, _ = self.find_rho_batch(p_input, T, P, rho_init=rho_init,
                                            n_steps=self.transport_rho_n_steps)
            rho = rho.detach()

        x_in       = torch.cat([T, rho], dim=1)
        x_in_norm  = self.normalize_input(x_in)
        eos_result = self.forward_normalized(p_input, x_in_norm)

        a_res  = self._denormalize_a_res(eos_result[:, 0:1])
        s_res  = self._invert_grad_T( eos_result[:, 1:2], T, a_res)
        u_res  = a_res + T * s_res
        v_m    = 1.0 / rho.clamp(min=1e-6)

        # MotHONet: branch=substance params, trunk=[-s_res/R, u_res/RT]
        s_res_dimless = -s_res / R
        u_res_dimless = -u_res / (R * T)
        transport_in = torch.cat([p_input, s_res_dimless, u_res_dimless], dim=1)
        de_dv, neg_2_log_b, neg_log_df = self.transport_params(transport_in)

        # MotH analytic parts; neg_2_log_b = -2·log(b) and neg_log_df = -log(df) from branch
        RT    = R * T
        v13   = v_m ** (1.0 / 3.0)
        nav13 = (6.02214076e23) ** (1.0 / 3.0)
        cc      = torch.log(torch.sqrt(2 * torch.pi * mw * RT) / (v13 * v13) / nav13)
        ln_vis  = cc + de_dv + neg_2_log_b
        dd      = torch.log(nav13 * 1.380649e-23 * T / (6 * v13)) + neg_log_df
        ln_diff = dd - ln_vis

        result = torch.cat([eos_result, ln_vis, ln_diff], dim=1)
        return result, x_in, a_res, s_res, None


    def forward_transport(self, substance_ids, T, P, mw, rho_init=None):
        """Inference wrapper around forward_transport_train."""
        result, x_in, a_res, s_res, _ = self.forward_transport_train(substance_ids, T, P, mw, rho_init)
        transport_preds = result[:, 3:]   # [ln_vis, ln_diff]
        rho = x_in[:, 1:2]
        return transport_preds, rho, (a_res, s_res)

    # =========================================================================
    # Helmholtz freeze / rho cache (used during transport pre-training)
    # =========================================================================

    @property
    def _helmholtz_is_frozen(self) -> bool:
        return not any(
            p.requires_grad
            for n, p in self.named_parameters()
            if 'transport_params' not in n
        )

    def freeze_helmholtz(self):
        for n, p in self.named_parameters():
            if 'transport_params' not in n:
                p.requires_grad_(False)
        self._rho_cache = None

    def freeze_helmholtz_layers(self,not_frozen_layers=[0]):
        for n, p in self.named_parameters():
            if 'transport_params' not in n:
                nf_flag = False
                for nfl in not_frozen_layers:
                    if f".{nfl}." in n:
                        nf_flag = True
                if not nf_flag:
                    p.requires_grad_(False)
        self._rho_cache = None        

    def unfreeze_helmholtz(self):
        for n, p in self.named_parameters():
            if 'transport_params' not in n:
                p.requires_grad_(True)
        self._rho_cache = None

    def _rho_cache_lookup(self, substance_ids, T, P):
        if self._rho_cache is None:
            return None
        keys = torch.stack([substance_ids.float(), T.squeeze(1), P.squeeze(1)], dim=1)
        match = (self._rho_cache[:, :3].unsqueeze(0) == keys.unsqueeze(1)).all(dim=2)
        if not match.any(dim=1).all():
            return None
        idx = match.float().argmax(dim=1)
        return self._rho_cache[idx, 3:4].to(T.device)

    def _rho_cache_append(self, substance_ids, T, P, rho):
        new_rows = torch.stack(
            [substance_ids.float(), T.squeeze(1), P.squeeze(1), rho.squeeze(1)], dim=1
        ).detach().cpu()
        self._rho_cache = (
            torch.cat([self._rho_cache, new_rows], dim=0)
            if self._rho_cache is not None else new_rows
        )

    def find_vle_flash_batch_norm(self, substance_ids, x_in,
                            n_steps=20, tol=1e-5, alpha=0.5, gas_reset_factor=0.95,
                            liq_reset_factor=1.05, via_gradient2_rhorho=False, h_rel=1e-2,
                            convergence_threshold=1000.0):
        """
        Batch flash calculation using total pressure and chemical potential.
        Returns normalized saturation properties and convergence mask (use in training).

        Parameters:
        -----------
        via_gradient2_rhorho : bool
            If True, use autograd second derivatives from model.forward()
            If False, compute dp/dρ and dμ/dρ via finite differences (recommended)
        h_rel : float
            Relative step size for finite differences (only used if via_gradient2_rhorho=False)
        convergence_threshold : float
            Maximum |p_gas - p_liq| for a sample to be considered converged (default: 1000 Pa)
        """
        print("DEPRECIATED, USED IN TRAINING BUT SHIT")
        p_input = self.substance_params(substance_ids)

        T_batch = x_in[:, 0:1]
        rho_gas_init = x_in[:, 1:2]
        rho_liquid_init = x_in[:, 2:3]

        # get VLE flash results including convergence mask
        result = self.find_vle_flash_batch(
            p_input, T_batch, rho_gas_init, rho_liquid_init,
            n_steps=n_steps, tol=tol, alpha=alpha, gas_reset_factor=gas_reset_factor,
            liq_reset_factor=liq_reset_factor, via_gradient2_rhorho=via_gradient2_rhorho, h_rel=h_rel,
            convergence_threshold=convergence_threshold
        )
        p_sat_pred, mu_sat_pred, _, _, converged_mask = result
        p_sat_norm = self._normalize_p_sat(p_sat_pred)
        mu_sat_norm = self._normalize_mu_sat(mu_sat_pred)
        return torch.cat([p_sat_norm, mu_sat_norm], dim=1), converged_mask 


    def find_vle_flash_batch_train(self, substance_ids, x_in,
                            n_steps=50, tol=1e-5, alpha=0.5, gas_reset_factor=0.95,
                            liq_reset_factor=1.05, via_gradient2_rhorho=False, h_rel=1e-2,
                            convergence_threshold=1000.0):
        """
        VLE flash for training: Newton loop (detached) to find liquid density, then
        forward_train at that density WITH gradients.

        Unlike find_vle_flash_batch_norm, this avoids catastrophic cancellation by
        treating p_res at the VLE density as the training target, not log(p_sat).
        The training loss should use _extract_grad_rho_from_p_res with:
            p_res_sat_target = p_sat_ref - R * T * rho_liq_conv

        Returns:
            returns_vle : forward_train output at converged liquid density, shape (batch, 4)
                          columns: [a_res_norm, grad_T_norm, grad_rho_norm, grad_rho_norm]
            converged_mask : bool tensor, shape (batch, 1)
            rho_liq_conv : converged liquid density (detached), shape (batch, 1)
        """
        T_batch = x_in[:, 0:1]
        rho_gas_init = x_in[:, 1:2]
        rho_liquid_init = x_in[:, 2:3]
        # 4th column: 0 = rho_liq fixed (known), 1 = rho_liq free; default all-ones = free
        determine_rho_liq = x_in[:, 3:4] if x_in.shape[1] > 3 else torch.ones_like(T_batch)

        p_input = self.substance_params(substance_ids)
        _, _, rho_gas_conv, rho_liq_conv, converged_mask = self.find_vle_flash_batch(
            p_input, T_batch, rho_gas_init, rho_liquid_init,
            n_steps=n_steps, tol=tol, alpha=alpha,
            gas_reset_factor=gas_reset_factor, liq_reset_factor=liq_reset_factor,
            via_gradient2_rhorho=via_gradient2_rhorho, h_rel=h_rel,
            convergence_threshold=convergence_threshold,
            determine_rho_liq=determine_rho_liq,
        )

        phase_ok = (rho_liq_conv / rho_gas_conv.clamp(min=1e-6) >= 5.0) & torch.isfinite(rho_liq_conv)
        converged_mask = converged_mask & phase_ok

        x_vle_liq = torch.cat([T_batch, rho_liq_conv], dim=1)
        returns_liq = self.forward_train(substance_ids, x_vle_liq, compute_gradients=True)

        x_vle_gas = torch.cat([T_batch, rho_gas_conv], dim=1)
        returns_gas = self.forward_train(substance_ids, x_vle_gas, compute_gradients=True)

        return returns_liq, returns_gas, x_vle_liq, x_vle_gas, converged_mask


    def find_vle_curve(self, p_input, T_array, rho_liq_init, rho_gas_init, n_steps=40):
        """
        Saturation curve by continuation from T_max downward.

        Newton flash warm-started from previous converged (rho_liq, rho_gas).
        ΔH_vap and p_sat_cc computed post-loop in batch (not in the loop).

        Parameters
        ----------
        p_input      : Tensor (1, p_dim) — substance embedding
        T_array      : Tensor (n_T,) — temperatures, any order (sorted high→low internally)
        rho_liq_init : float — initial liquid density at T_max (mol/m³)
        rho_gas_init : float — initial vapor density at T_max (mol/m³)
        n_steps      : int — Newton steps per temperature (default 40)

        Returns (all n_T tensors, sorted T descending)
        -------
        T_out, p_sat_out, p_sat_cc_out, rho_liq_out, rho_gas_out,
        delta_H_vap_out, converged_out
          p_sat_out       — Newton-converged saturation pressure
          p_sat_cc_out    — CC baseline anchored at highest-T converged point
          delta_H_vap_out — thermodynamic ΔH_vap from model a_res, s_res (J/mol)
        """
        R = 8.314
        T_sorted = torch.sort(T_array.float(), descending=True).values

        n = len(T_sorted)
        T_out = T_sorted
        # columns: 0=p_sat, 1=mu_sat, 2=rho_gas, 3=rho_liq, 4=converged
        out = torch.empty(n, 5, dtype=torch.float32)

        rho_liq, rho_gas = float(rho_liq_init), float(rho_gas_init)

        # Pre-scan: step down from T_max until the flash converges.
        # If the initial T is near the critical point, convergence may fail and
        # cascade bad warm-starts to all lower temperatures. Find an anchor first.
        i_anchor = 0
        for i_anchor, T_val in enumerate(T_sorted):
            T_b   = T_val.reshape(1, 1)
            rho_g = torch.tensor([[rho_gas]], dtype=torch.float32)
            rho_l = torch.tensor([[rho_liq]], dtype=torch.float32)
            result = self.find_vle_flash_batch(p_input, T_b, rho_g, rho_l, n_steps=n_steps)
            out[i_anchor] = torch.cat(result, dim=1).squeeze()
            if out[i_anchor, 4].item():
                rho_liq = out[i_anchor, 3].item()
                rho_gas = out[i_anchor, 2].item()
                break  # found converging anchor

        for i in range(i_anchor + 1, n):
            T_b   = T_sorted[i].reshape(1, 1)
            rho_g = torch.tensor([[rho_gas]], dtype=torch.float32)
            rho_l = torch.tensor([[rho_liq]], dtype=torch.float32)

            result = self.find_vle_flash_batch(p_input, T_b, rho_g, rho_l, n_steps=n_steps)
            out[i] = torch.cat(result, dim=1).squeeze()

            if out[i, 4].item():
                rho_liq = out[i, 3].item()
                rho_gas = out[i, 2].item()

        p_out  = out[:, 0]
        rg_out = out[:, 2]
        rl_out = out[:, 3]
        cv_out = out[:, 4].bool()

        # Post-loop: batch ΔH_vap — two forward passes over all points
        x_liq = torch.stack([T_out, rl_out], dim=1)  # (n, 2)
        x_gas = torch.stack([T_out, rg_out], dim=1)  # (n, 2)
        p_exp = p_input.expand(n, -1)                 # (n, p_dim)

        a_liq, s_liq, _, _ = self.forward(p_exp, x_liq)  # (n, 1) each
        a_gas, s_gas, _, _ = self.forward(p_exp, x_gas)

        # ΔH_vap = Δa_res + T·Δs_res + p_sat·(1/rho_gas − 1/rho_liq)
        # Derived from H = A + TS + p/ρ; ideal-gas terms cancel at same T.
        # .squeeze(1): forward returns (n,1); T_out/p_out are (n,)
        dHvap_out = ((a_gas - a_liq).squeeze(1)
                     + T_out * (s_gas - s_liq).squeeze(1) 
                     # + R*T_out )
                     + p_out * (1.0 / rg_out - 1.0 / rl_out))
        # dHvap_out = torch.where(cv_out, dHvap_out,
        #                        torch.full_like(dHvap_out, float('nan')))

        # Post-loop: p_sat_cc anchored at highest-T converged point using model ΔH_vap
        def grid_and_extend(x):
            x_min = 2 * torch.min(x) - x
            x_max = 2 * torch.max(x) - x
            return torch.cat([x_min, x[1:-1], x_max])
        temperature_extend  = grid_and_extend(T_out)
        pressure_extend     = grid_and_extend(p_out)
        delta_H_vap_extend  = grid_and_extend(dHvap_out)  

        pp = temperature_extend > 0
        temperature_extend = temperature_extend[pp]
        pressure_extend    = pressure_extend[pp]
        delta_H_vap_extend = delta_H_vap_extend[pp]

        # (n, N_ext) — rows = target temperatures, cols = source temperatures
        t_diff  = torch.abs(T_out[:, None] - temperature_extend[None, :])
        weights = 1 / (1 + torch.exp(t_diff / 10))
        weights /= weights.sum(axis=1, keepdims=True)

        p_sat_cc = (pressure_extend[None, :]
                        * torch.exp(delta_H_vap_extend[None, :] / 8.314
                                * (1/temperature_extend[None, :] - 1/T_out[:, None]))
                        * weights).sum(axis=1)

        return T_out, p_out, p_sat_cc, rl_out, rg_out, dHvap_out, cv_out


    # ========== HELPER: NORMALIZE/DENORMALIZE A_RES ==========
    def _normalize_a_res(self, a_res):
        """Normalize log(-a_res) to [0, 1] using stored bounds."""
        a_res = torch.clamp(a_res, max=-1e-10)
        return torch.log(-a_res)  # a_res is negative

    def _denormalize_a_res(self, a_log):
        """Inverse: normalized → log(-a_res) → a_res"""
        # a_res = torch.clamp(-torch.exp(a_log), max=-10.0)
        a_res = torch.clamp(-torch.exp(a_log), max=-1e-6)  # no magnitude floor
        return a_res

    def _normalize_p_sat(self, p_sat):
        """Normalize log(p_sat) to [0, 1] using stored bounds."""
        print("DEPRECIATED, USED IN TRAINING BUT SHIT")
        p_sat = torch.clamp(p_sat, min=1e-10)
        return torch.log(p_sat)  # p_sat is positive

    def _denormalize_p_sat(self, p_log):
        """Inverse: normalized → log(p_sat) → p_sat"""
        print("DEPRECIATED, USED IN TRAINING BUT SHIT")
        return torch.exp(p_log)  # Positive sign to get p_sat

    def _normalize_mu_sat(self, mu_sat):
        """Normalize log(mu_sat) to [0, 1] using stored bounds."""
        print("DEPRECIATED, USED IN TRAINING BUT SHIT")
        mu_sat = torch.clamp(mu_sat, min=1e-10)
        return torch.log(mu_sat)  # mu_sat is positive

    def _denormalize_mu_sat(self, mu_log):
        """Inverse: normalized → log(mu_sat) → mu_sat"""
        print("DEPRECIATED, USED IN TRAINING BUT SHIT")
        return torch.exp(mu_log)  # Positive sign to get mu_sat

    # ========== HELPER: NORMALIZE/DENORMALIZE TRANSPORT ==========
    def _normalize_viscosity(self, vis):
        """Normalize vis"""
        vis = torch.clamp(vis, min=1e-10)
        return torch.log(vis)

    def _denormalize_viscosity(self, vis_ln):
        """Inverse: normalized → log(vis) → vis"""
        return torch.exp(vis_ln)

    def _normalize_diffusion(self, d):
        """Normalize d"""
        d = torch.clamp(d, min=1e-10)
        return torch.log(d)

    def _denormalize_diffusion(self, d_ln):
        """Inverse: normalized → log(d) → d"""
        return torch.exp(d_ln)
    
    # ========== HELPER: INPUT/OUTPUT DE/NORMALIZATION ==========
    def _denormalize_T(self, T_norm):
        """Denormalize log(T) from normalized space."""
        return torch.exp(T_norm)

    def _denormalize_rho(self, rho_norm):
        """Denormalize log(rho) from normalized space."""
        return torch.exp(rho_norm)

    def _normalize_T(self, T):
        """Normalize log(T) to [0, 1] using stored bounds."""
        T = torch.clamp(T, min=1e-8)
        return torch.log(T) 

    def _normalize_rho(self, rho):
        """Normalize log(rho) to [0, 1] using stored bounds."""
        rho = torch.clamp(rho, min=1e-8)
        return torch.log(rho) 
    
    def _denormalize_transport(self, transport_norm):
        """Denormalize transport property
            [viscosity, diffusion, ...]
            log(tp) from normalized space."""
        return torch.exp(transport_norm)

    def _normalize_transport(self, T):
        """Normalize transport property
            [viscosity, diffusion, ...]
            log(tp) from normalized space."""
        transport_norm = torch.clamp(transport_norm, min=1e-8)
        return torch.log(transport_norm) 

    # ========== HELPER: GRADIENT EXTRACTION FROM LOG-RESIDUALS ==========
    def _extract_grad_T_from_s_res(self, s_res, T, a_res):
        """
        Extract grad_T from s_res.
        """
        return -s_res * T  /  a_res

    def _extract_grad_rho_from_p_res(self, p_res, rho, a_res):
        """
        Extract grad_rho from p_res.
        """
        return p_res / (rho * a_res)

    def _extract_grad_rho_from_mu_res(self, mu_res, a_res):
        """
        Extract grad_rho from mu_res.
        """
        return (mu_res - a_res) / a_res

    def _extract_grad_T_from_dh_vap(self, dh_vap, returns_liq_norm, returns_gas_norm, x_vle_liq, x_vle_gas):
        """
        Extract grad_T from dH_vap.
        Input: vle train returns + dhvap
        """
        # x_vle_liq / x_vle_gas are in physical units [T_K, rho_mol_m3] — use directly.
        # residuals_from_gradients expects normalized x_in, so normalize before calling.
        T       = x_vle_liq[:, 0:1]   # K
        rho_liq = x_vle_liq[:, 1:2]   # mol/m³
        rho_gas = x_vle_gas[:, 1:2]   # mol/m³

        x_vle_liq_norm = self.normalize_input(x_vle_liq)
        x_vle_gas_norm = self.normalize_input(x_vle_gas)

        res_liq = self.residuals_from_gradients(
                            returns_liq_norm[:,0:1], returns_liq_norm[:,1:2],
                            returns_liq_norm[:,2:3], x_vle_liq_norm
                        )
        a_res_liq, s_res_liq, p_res_liq, mu_res_liq = res_liq
        p_liq = p_res_liq + 8.314 * T * rho_liq

        res_gas = self.residuals_from_gradients(
                            returns_gas_norm[:,0:1], returns_gas_norm[:,1:2],
                            returns_gas_norm[:,2:3], x_vle_gas_norm
                        )
        a_res_gas, s_res_gas, p_res_gas, mu_res_gas = res_gas
        p_gas = p_res_gas + 8.314 * T * rho_gas
        
        da_vap = a_res_gas - a_res_liq
        pdv_vap = p_liq * (1/rho_gas - 1/rho_liq)
        tds_vap = T * ( s_res_gas - s_res_liq )
        dummy = ( dh_vap - da_vap - pdv_vap - T*s_res_gas ) / a_res_liq
        dummy = (dh_vap - T*s_res_gas - 8.314 * T * torch.log(rho_liq / rho_gas)) / (a_res_liq)
        #dummy0 = dh_vap - pdv_vap - tds_vap - a_res_gas
        #dummy0 = self._normalize_a_res(-dummy0)
        #print(dh_vap.mean())
        #print(a_res_gas.mean(), a_res_liq.mean(), da_vap.mean())
        #print(pdv_vap.mean(), tds_vap.mean())
        #print(dummy.mean(), returns_liq_norm[:,1:2].mean())
        #print(dummy0.mean(), returns_liq_norm[:,0:1].mean())
        #print()
        return dummy

    # ========== HELPER: A_RES EXTRACTION FROM LOG-RESIDUALS ==========
    def _extract_a_res_from_s_res(self, s_res, T, grad_T):
        """
        Extract a_res from s_res.
        """
        return -s_res * T  /  grad_T

    def _extract_a_res_from_p_res(self, p_res, rho, grad_rho):
        """
        Extract a_res from p_res.
        """
        return p_res / (rho * grad_rho)
    
    def _extract_a_res_from_mu_res(self, mu_res, grad_rho):
        """
        Extract a_res from mu_res.
        """
        return mu_res / (1 + grad_rho)    

    # ========== HELPER: INVERSE GRADIENT EXTRACTION ==========
    def _invert_grad_T(self, grad_T_norm, T, a_res):
        """Inverse: grad_T_norm → s_res"""
        return -a_res / T * grad_T_norm

    def _invert_grad_rho_p(self, grad_rho_norm, rho, a_res):
        """Inverse: grad_rho_norm → p_res"""
        return a_res  * rho * grad_rho_norm

    def _invert_grad_rho_mu(self, grad_rho_norm, a_res):
        """Inverse: grad_rho_norm → mu_res"""
        return a_res + ( a_res ) * grad_rho_norm

    def _ideal_gas_dp_drho(self, T):
        """Ideal gas contribution to dp/drho: RT."""
        return 8.314 * T

    def _ideal_gas_dmu_drho(self, T, rho):
        """Ideal gas contribution to dmu/drho: RT/rho."""
        return 8.314 * T / rho

    def _invert_grad_rho_rho_p(self, grad_rho_norm, grad_rho_rho_norm, a_res):
        """dp_res/drho from second normalized gradient (residual part only)."""
        g, h = grad_rho_norm, grad_rho_rho_norm
        return a_res * (g + g * g + h)

    def _invert_grad_rho_rho_mu(self, grad_rho_norm, grad_rho_rho_norm, rho, a_res):
        """dmu_res/drho from second normalized gradient (residual part only)."""
        g, h = grad_rho_norm, grad_rho_rho_norm
        return (a_res / rho) * (g + g * g + h)

    # ========== HELPER: GET RESIDUALS ==========
    def _residual_pressure(self, p, T, rho):
        """residual pressure from pressure """
        p_ig = 8.314*T*rho
        return p - p_ig

    def _residual_chemical_potential(self, mu, T, rho):
        """residual mu from mu """
        mu_ig = 8.314*T* torch.log(rho)
        return mu - mu_ig

    def estimate_lipschitz_constant(self) -> dict:
        """
        Estimate the Lipschitz constant of the network as an upper bound.

        Computes L ≤ ∏ σ_max(Wᵢ) × L_act, where σ_max is the largest
        singular value of each Linear layer's weight matrix and L_act is
        the Lipschitz constant of the activation function.

        This is a loose upper bound — the true constant is typically smaller
        because worst-case directions across layers rarely align.

        Returns
        -------
        dict with keys:
            'lipschitz_bound'   : float — the product bound L
            'activation_factor' : float — Lipschitz constant of the activation
            'spectral_norms'    : list[float] — per-layer σ_max values
        """
        activation_lipschitz = {
            "tanh": 1.0, "relu": 1.0, "elu": 1.0, "softplus": 1.0,
            "leaky_relu": 1.0, "sigmoid": 0.25,
            "gelu": 1.13, "silu": 1.13,
        }
        L_act = activation_lipschitz.get(self.activation_name.lower(), 1.0)

        spectral_norms = []
        with torch.no_grad():
            for module in self.simple_net.bnn.modules():
                if isinstance(module, nn.Linear):
                    sigma = torch.linalg.matrix_norm(module.weight, ord=2).item()
                    spectral_norms.append(sigma)

        L = L_act ** (len(spectral_norms) - 1)  # activation applied between layers
        for sigma in spectral_norms:
            L *= sigma

        return {
            "lipschitz_bound": L,
            "activation_factor": L_act,
            "spectral_norms": spectral_norms,
        }

