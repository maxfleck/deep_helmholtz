from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from helmholtz_eos_net.dataset import ListedDataset


def compute_adaptive_weights_normalized(losses, ema_losses,
                                        min_weight_floors,
                                        initial_losses=None,
                                        ema_alpha=0.1,
                                        max_weight_ceilings=None):
    """
    Compute adaptive weights using normalized loss ratios (vectorized).

    Parameters:
    -----------
    losses : torch.Tensor
        Current loss values [n]
    ema_losses : torch.Tensor
        Running EMA state [n]
    initial_losses : torch.Tensor or None
        Initial loss values for normalization. If None, will be set on first call.
    ema_alpha : float
        EMA smoothing factor (default 0.1)

    Returns:
    --------
    weights : torch.Tensor
        Computed weights [n]
    ema_losses : torch.Tensor
        Updated EMA state
    initial_losses : torch.Tensor
        Stored initial losses
    """
    # Initialize tracking on first call
    if initial_losses is None:
        initial_losses = torch.clamp(losses, min=1e-8)

    # Update EMA (vectorized)
    ema_losses = (1 - ema_alpha) * ema_losses + ema_alpha * losses

    # Compute relative loss ratios
    loss_ratios = ema_losses / initial_losses

    # Compute inverse weights
    eps = 1e-10
    weights = 1.0 / (loss_ratios + eps)

    # Normalize weights to sum to n (so average weight is 1.0)
    n = len(losses)
    weights = n * weights / weights.sum()

    # Apply minimum weight floors
    weights = torch.max(torch.stack([weights, min_weight_floors]), dim=0).values

    # Apply maximum weight ceilings (for "side quest" goals that shouldn't dominate)
    if max_weight_ceilings is not None:
        weights = torch.min(torch.stack([weights, max_weight_ceilings]), dim=0).values

    return weights, ema_losses, initial_losses


class HEOS_forward_train_orchestrator():
        
    def __init__(self, model, config, return_keys):
    
        self.model = model
        self.model_index = 0,
        self.model_name = "HEOS_forward_train"
        self.model_forward_method="forward_train",
        self.model_forward= ["temperature_K", "molar_density_mol_m3"]
        self.model_kwargs = ["compute_gradients", "compute_gradient2_rhorho"]
        self.model_return = ["helmholtz_energy_res_J_mol", "entropy_res_J_K",
                            "pressure_res_Pa", "chemical_potential_res_J_mol"]

        # Map each goal key to the model output column it reads from
        self.return_indexes = [GOAL_REGISTRY[key].output_index for key in return_keys]
        self.return_keys = return_keys
        self.target_indexes = list(range(len(return_keys)))

        self.compute_gradients = any(GOAL_REGISTRY[key].requires_gradients for key in return_keys)
        self.compute_gradient2_rhorho = False

        self.config = config
        return

    def get_goals(self, x_in, returns, y_targets):
        # Detach all model outputs used in target computations: targets are treated as
        # snapshots of the current model state, not as a gradient path. Each loss trains
        # exactly the output indexed by return_indexes, not the outputs used in the target
        # formula. This avoids gradient interference and moving-target non-stationarity.
        # returns = returns.detach()

        a_res_norm = returns[:,0:1]
        # grad_T_norm = returns[:,1:2]
        # grad_rho_norm = returns[:,2:3]

        # Denormalize inputs
        T = x_in[:, 0:1]
        rho = x_in[:, 1:2]

        # Denormalize primary output
        a_res = self.model._denormalize_a_res(a_res_norm)

        # Level 2 (Inner): Convert normalized gradients to log-residuals
        # s_res = self.model._invert_grad_T(grad_T_norm, T, a_res)
        # p_res = self.model._invert_grad_rho_p(grad_rho_norm, rho, a_res)
        # mu_res = self.model._invert_grad_rho_mu(grad_rho_norm, a_res)

        goals = []
        for i,key in enumerate(self.return_keys):
            if key == "helmholtz_energy_res_J_mol":
                goal = self.model._normalize_a_res(y_targets[:,i:i+1])
            elif key == "entropy_res_J_K":
                goal = self.model._extract_grad_T_from_s_res(y_targets[:,i:i+1], T, a_res)
            elif key == "pressure_res_Pa":
                goal = self.model._extract_grad_rho_from_p_res(y_targets[:,i:i+1], rho, a_res)
            elif key == "chemical_potential_res_J_mol":
                goal = self.model._extract_grad_rho_from_mu_res(y_targets[:,i:i+1], a_res)
            elif key == "a_res_from_p_res":
                grad_rho = returns[:, 2:3]
                a_res_p = self.model._extract_a_res_from_p_res(y_targets[:,i:i+1], rho, grad_rho)
                goal = self.model._normalize_a_res(a_res_p)
            else:
                print(f"WARNING: key {key} not in model returns")
            goals.append(goal)

        goals = torch.cat(goals, dim=1)
        return torch.nan_to_num(goals, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50, 50)

    def forward(self, substance_ids, x_in, y_targets, return_jacobian=False):
        # Derive valid mask before get_goals() converts NaN to 0
        valid_mask = torch.isfinite(y_targets)   # (batch, n_goals)

        returns = self.model.forward_train(substance_ids, x_in,
                                    compute_gradients=self.compute_gradients,
                                    compute_gradient2_rhorho=self.compute_gradient2_rhorho,
                                    )
        if self.compute_gradients:
            predictions = returns[:,self.return_indexes]
            goals = self.get_goals( x_in, returns, y_targets)
            # NaN/Inf predictions → replace with goal so loss=0 (not NaN)
            outlier_mask = ~torch.isfinite(predictions)
            predictions = torch.where(outlier_mask, goals, predictions)
            if return_jacobian:
                jacobian_norm = returns[:, 1:3].norm(dim=1)
                return predictions, goals, valid_mask, jacobian_norm
        else:
            predictions = returns
            goals = self.model._normalize_a_res(y_targets[:,0:1])
        return predictions, goals, valid_mask


class HEOS_vle_norm_orchestrator():

    def __init__(self, model, config, return_keys):
    
        self.model = model
        self.model_index = 1,
        self.model_name = "HEOS_vle_norm"
        self.model_forward_method="find_vle_flash_batch_norm",
        self.model_forward= ["temperature_K", "molar_density_vapor_mol_m3",
                              "molar_density_liquid_mol_m3", "determine_rho_liq"]
        self.model_kwargs = []
        # Column layout matches forward_train: [a_res_norm(0), grad_T(1), grad_rho_p(2), grad_rho_mu(3)]
        self.model_return = ["helmholtz_energy_res_J_mol", "entropy_res_J_K",
                             "pressure_sat_Pa_liq", "chemical_potential_sat_J_mol"]

        # Separate data-backed goals from self-supervised equilibrium constraints
        _EQUIL_KEYS = {"ln_fug_equil"}
        self.data_return_keys = [k for k in return_keys if k not in _EQUIL_KEYS]
        self.equil_return_keys = [k for k in return_keys if k in _EQUIL_KEYS]

        # Map each data goal key to the VLE model output column it reads from
        # (equil keys use output_index=-1 sentinel and are handled separately)
        self.return_indexes = [GOAL_REGISTRY[key].output_index for key in self.data_return_keys]
        self.return_keys = self.data_return_keys   # get_goals() iterates this
        self.target_indexes = list(range(len(self.data_return_keys)))

        self.config = config
        self.use_vle_search = config.get("use_vle_search", False)
        self.vle_search_n_steps = config.get("vle_search_n_steps", 10)
        return

    def get_goals(self, returns_liq, x_liq, returns_vap, x_vap, y_targets):
        # Detach all model outputs used in target computations: targets are treated as
        # snapshots of the current model state, not as a gradient path. Each loss trains
        # exactly the output indexed by return_indexes, not the outputs used in the target
        # formula. This avoids gradient interference and moving-target non-stationarity.
        # returns_liq = returns_liq.detach()
        # returns_vap = returns_vap.detach()

        a_res_norm = returns_liq[:, 0:1]
        T = x_liq[:, 0:1]
        rho = x_liq[:, 1:2]
        a_res = self.model._denormalize_a_res(a_res_norm)

        goals = []
        for i, key in enumerate(self.return_keys):
            if key == "pressure_sat_Pa_liq":
                p_res = self.model._residual_pressure(y_targets[:, i:i+1], T, rho)
                goal = self.model._extract_grad_rho_from_p_res(p_res, rho, a_res)
            elif key == "pressure_sat_Pa_gas":
                rho_vap = x_vap[:, 1:2]
                p_res = self.model._residual_pressure(y_targets[:, i:i+1], T, rho_vap)
                goal = self.model._extract_grad_rho_from_p_res(p_res, rho_vap, a_res)
            elif key == "chemical_potential_sat_J_mol":
                mu_res = self.model._residual_chemical_potential(y_targets[:, i:i+1], T, rho)
                goal = self.model._extract_grad_rho_from_mu_res(mu_res, a_res)
            elif key == "molar_vaporization_enthalpy_J_mol":
                goal = self.model._extract_grad_T_from_dh_vap(
                    y_targets[:, i:i+1], returns_liq, returns_vap, x_liq, x_vap
                )
            else:
                print(f"WARNING: key {key} not in model returns")
            goals.append(goal)

        goals = torch.cat(goals, dim=1)
        return torch.nan_to_num(goals, nan=0.0, posinf=0.0, neginf=0.0).clamp(-50, 50)

    def forward(self, substance_ids, x_in, y_targets, return_jacobian=False):
        # Derive valid mask before get_goals() converts NaN to 0.
        # Equil keys have no y_targets column → always valid (padded with ones).
        valid_mask_data = torch.isfinite(y_targets)  # (batch, n_data_goals)
        if self.equil_return_keys:
            valid_mask_equil = torch.ones(
                len(y_targets), len(self.equil_return_keys),
                dtype=torch.bool, device=y_targets.device
            )
            valid_mask = torch.cat([valid_mask_data, valid_mask_equil], dim=1)
        else:
            valid_mask = valid_mask_data

        T = x_in[:, 0:1]
        rho_vap = x_in[:, 1:2]
        rho_liq = x_in[:, 2:3]
        n_temps = len(T)

        if self.use_vle_search:
            # print("VLE flashhhh")
            # VLE flash runs detached; forward_train at converged densities has gradients.
            # x_in shape [T, rho_gas_init, rho_liq_init] matches find_vle_flash_batch_train input.
            returns_liq, returns_vap, x_in_liq, x_in_vap, _ = \
                self.model.find_vle_flash_batch_train(
                    substance_ids, x_in, n_steps=self.vle_search_n_steps
                )
            rho_liq = x_in_liq[:, 1:2]
            rho_vap = x_in_vap[:, 1:2]
        else:
            #print("noooo VLE flashhhh")
            T_merged = torch.cat([T, T])
            rho_merged = torch.cat([rho_vap, rho_liq], dim=0)
            x_in_merged = torch.cat([T_merged, rho_merged], dim=1)
            substance_ids_merged = torch.cat([substance_ids, substance_ids])
            returns = self.model.forward_train(substance_ids_merged, x_in_merged, compute_gradients=True)
            returns_vap = returns[:n_temps, :]
            returns_liq = returns[n_temps:, :]
            x_in_liq = torch.cat([T, rho_liq], dim=1)
            x_in_vap = torch.cat([T, rho_vap], dim=1)

        # Reconstruct merged tensors for ln_p / ln_fug computations
        T_merged = torch.cat([T, T])
        rho = torch.cat([rho_vap, rho_liq], dim=0)
        returns = torch.cat([returns_vap, returns_liq], dim=0)

        preds_liq = returns_liq[:, self.return_indexes]

        # Convert to total pressure and chemical potential/fugacity
        y_norm = returns[:, 0:1]
        grad_rho_norm = returns[:, 2:3]

        a_res = self.model._denormalize_a_res(y_norm)
        p_res = self.model._invert_grad_rho_p(grad_rho_norm, rho, a_res)
        mu_res = self.model._invert_grad_rho_mu(grad_rho_norm, a_res)

        p = p_res + 8.314 * T_merged * rho
        ln_p = torch.log(p)
        mu = mu_res + 8.314 * T_merged * torch.log(torch.clamp(rho, min=1e-6))
        mu = mu + 8.314 * T_merged * torch.log(8.314 * T_merged)
        ln_fug = torch.log(torch.clamp(mu / (8.314 * T_merged), min=1.0))
        ln_p_vap = ln_p[:n_temps, :]
        ln_p_liq = ln_p[n_temps:, :]
        ln_fug_vap = ln_fug[:n_temps, :]
        ln_fug_liq = ln_fug[n_temps:, :]

        goals_liq = self.get_goals(returns_liq, x_in_liq, returns_vap, x_in_vap, y_targets)

        goals_liq = torch.nan_to_num(goals_liq, nan=0.0, posinf=0.0, neginf=0.0)
  
        # Mask each phase independently, then average — keeps (batch, n_keys) shape
        # not_converged = ~converged_mask
        # mask_liq = not_converged | ((preds_liq - goals_liq).abs() > 2.0)
        # Outlier mask only — convergence criterion (absolute Pa) is too strict at low p_sat
        # mask_liq = ~torch.isfinite(preds_liq) | ((preds_liq - goals_liq).abs() > 2.0)
        mask_liq = ~torch.isfinite(preds_liq)
        # mask_liq = ((preds_liq - goals_liq).abs() > 2.0)

        preds_liq = torch.where(mask_liq, goals_liq, preds_liq)

        #predictions = (preds_liq + preds_gas) / 2.0
        #goals       = (goals_liq + goals_gas) / 2.0

        predictions = preds_liq
        goals       = goals_liq

        if self.equil_return_keys:
            equil_preds, equil_goals = [], []
            for key in self.equil_return_keys:
                if key == "log_p_consistency":
                    # ln(p_vap/p_liq) = ln_p_vap - ln_p_liq → 0 at equilibrium
                    diff = torch.nan_to_num(ln_p_vap - ln_p_liq, nan=0., posinf=0., neginf=0.)
                    equil_preds.append(diff)
                    equil_goals.append(torch.zeros_like(diff))
                elif key == "ln_fug_equil":
                    diff = torch.nan_to_num(ln_fug_vap - ln_fug_liq, nan=0., posinf=0., neginf=0.)
                    equil_preds.append(diff)
                    equil_goals.append(torch.zeros_like(diff))
            predictions = torch.cat([predictions] + equil_preds, dim=1)
            goals       = torch.cat([goals]       + equil_goals, dim=1)

        if return_jacobian:
            return predictions, goals, valid_mask, returns_liq[:, 1:3].norm(dim=1)
        return predictions, goals, valid_mask

class HEOS_transport_orchestrator:
    """
    Orchestrator for transport property training (viscosity, diffusion).

    Calls model.forward_transport(substance_ids, T, P, mw) which:
      1. Finds density via find_rho_batch (Newton, detached)
      2. Evaluates EOS for a_res, s_res at converged density
      3. Applies MotH functional forms with transport_params (b, n, m, df)

    Input CSV columns required: temperature_K, pressure_Pa, molar_mass_kg_mol
    Optional initial density: molar_density_mol_m3
    """

    def __init__(self, model, config, return_keys):
        self.model = model
        self.model_index = 2
        self.model_name = "HEOS_transport"
        self.model_forward = ["temperature_K", "pressure_Pa", "molar_mass_kg_mol",
                               "molar_density_mol_m3"]   # last col used as rho_init
        self.model_kwargs = []
        self.return_keys = return_keys
        self.return_indexes = [GOAL_REGISTRY[k].output_index for k in return_keys]
        self.target_indexes = list(range(len(return_keys)))
        self.config = config

    def get_goals(self, y_targets, a_res, T_phys, m_param, contributions):
        """Build per-key goals: measured ln(transport) as direct targets."""
        R = 8.31446261815324
        RT = R * T_phys
        goals = []
        for i, key in enumerate(self.return_keys):
            y_col = y_targets[:, i:i+1]
            if key == "viscosity_Pa_s":
                vis_safe = y_col.nan_to_num(nan=1.0).clamp(min=1e-30)
                goal = self.model._normalize_viscosity(vis_safe)
            elif key == "diffusion_m2_s":
                diff_safe = y_col.nan_to_num(nan=1e-9).clamp(min=1e-200)
                goal = self.model._normalize_diffusion(diff_safe)
            else:
                print(f"WARNING: key {key} not in model returns")
                goal = y_col
            goals.append(goal)
        # Clamp finite values to ±50; indirect goals can blow up near-zero a_res.
        return torch.nan_to_num(
            torch.cat(goals, dim=1), nan=float('nan'), posinf=50.0, neginf=-50.0
        ).clamp(-50, 50)

    def forward(self, substance_ids, x_in, y_targets, return_jacobian=False):
        # x_in columns: [T, P, mw, rho_init (optional)]
        T   = x_in[:, 0:1]
        P   = x_in[:, 1:2]
        mw  = x_in[:, 2:3]
        rho_init = x_in[:, 3:4] if x_in.shape[1] > 3 else None

        valid_mask = torch.isfinite(y_targets)

        # result: (batch, 5) = [a_res_norm, grad_T_norm, grad_rho_norm, ln_vis, ln_diff]
        result, x_in_phys, a_res, m_p, contributions = self.model.forward_transport_train(
            substance_ids, T, P, mw, rho_init=rho_init
        )
        T_phys = x_in_phys[:, 0:1]

        predictions = result[:, self.return_indexes]
        goals = self.get_goals(y_targets, a_res, T_phys, m_p, contributions)

        outlier_mask = ~torch.isfinite(predictions)
        predictions = torch.where(outlier_mask, goals, predictions)

        return predictions, goals, valid_mask


# =============================================================================
# PRESETS
# =============================================================================
PRESETS = {
    'a_res_only': ["helmholtz_energy_res_J_mol"],
    'p_res_only': ["pressure_res_Pa"],
    'p_sat_only': ["pressure_sat_Pa_liq", "pressure_sat_Pa_gas"],
    'p_res_and_p_sat': ["pressure_res_Pa", "pressure_sat_Pa_liq", "pressure_sat_Pa_gas"],
    'all_residuals': [
        "helmholtz_energy_res_J_mol", "entropy_res_J_K",
        "pressure_res_Pa", "chemical_potential_res_J_mol"
    ],
    'all_plus_p_sat': [
        "helmholtz_energy_res_J_mol", "entropy_res_J_K",
        "pressure_res_Pa", "chemical_potential_res_J_mol",
        "pressure_sat_Pa_liq"
    ],
    'all_plus_sat': [
        "helmholtz_energy_res_J_mol", "entropy_res_J_K",
        "pressure_res_Pa", "chemical_potential_res_J_mol",
        "pressure_sat_Pa_liq", "chemical_potential_sat_J_mol"
    ],
    # a_res_from_p_res goals: train a_res via self-consistency with pressure data
    'p_res_with_a_res_p': ["pressure_res_Pa", "a_res_from_p_res"],
    'p_res_and_p_sat_with_a_res_p': ["pressure_res_Pa", "a_res_from_p_res", "pressure_sat_Pa_liq"],
    # dH_vap goals: train entropy (grad_T) from vaporization enthalpy data
    'dh_vap_only':                  ["molar_vaporization_enthalpy_J_mol"],
    'p_sat_and_dh_vap':             ["pressure_sat_Pa_liq", "pressure_sat_Pa_gas", "molar_vaporization_enthalpy_J_mol"],
    'p_res_and_p_sat_with_dh_vap':  ["pressure_res_Pa", "pressure_sat_Pa_liq", "pressure_sat_Pa_gas",
                                      "molar_vaporization_enthalpy_J_mol"],
    # transport goals — TransportNet trains directly on measured ln(transport)
    'transport_viscosity':    ["viscosity_Pa_s"],
    'transport_both':         ["viscosity_Pa_s", "diffusion_m2_s"],
    'p_res_and_transport_viscosity': ["pressure_res_Pa", "viscosity_Pa_s"],
    'p_res_and_transport_both':      ["pressure_res_Pa", "viscosity_Pa_s", "diffusion_m2_s"],
    'p_res_and_viscosity':      ["pressure_res_Pa", "viscosity_Pa_s"],
    'p_res_and_diffusion':      ["pressure_res_Pa", "diffusion_m2_s"],

}

# =============================================================================
# GOAL METADATA
# =============================================================================

@dataclass
class GoalMetadata:
    """
    Metadata describing a training goal's requirements and behavior.
    """
    model_index: int  # Which model to use in multi-model setups
    model_name: str
    output_index: int  # Index in forward_train output
    target_normalizer: str
    target_normalizer_args: List # Arguments idxs for target normalizer
    requires_gradients: bool
    requires_vle: bool
    target_key: str    # routing key used in PRESETS and get_goals(); may be virtual
    data_column: Optional[str]   # actual CSV column to load; None for self-supervised goals
    vle_init_columns: List[str] = field(default_factory=list)
    min_weight_floor: float = 0.01
    max_weight: float = 2.0
    ema_loss: float = 0.1

# =============================================================================
# GOAL REGISTRY
# =============================================================================

GOAL_REGISTRY: Dict[str, GoalMetadata] = {
    "helmholtz_energy_res_J_mol": GoalMetadata(
        model_index=0,
        model_name="HEOS_forward_train",
        target_normalizer="_normalize_a_res",
        target_normalizer_args=[[0],[],[]],
        requires_gradients=False,  # Can train a_res without gradients (Phase 1)
        requires_vle=False,
        target_key="helmholtz_energy_res_J_mol",
        data_column="helmholtz_energy_res_J_mol",
        vle_init_columns=[],
        output_index=0,
        min_weight_floor=0.1,
        ema_loss=0.1,
    ),
    "entropy_res_J_K": GoalMetadata(
        model_index=0,
        model_name="HEOS_forward_train",
        target_normalizer="_extract_grad_T_from_s_res",
        target_normalizer_args=[[1],[0],[0]],
        requires_gradients=True,  # Derived from dA/dT
        requires_vle=False,
        target_key="entropy_res_J_K",
        data_column="entropy_res_J_K",
        vle_init_columns=[],
        output_index=1,
        min_weight_floor=0.1,
        ema_loss=0.1,
    ),
    "pressure_res_Pa": GoalMetadata(
        model_index=0,
        model_name="HEOS_forward_train",
        target_normalizer="_extract_grad_rho_from_p_res",
        target_normalizer_args=[[2],[1],[0]],
        requires_gradients=True,  # Derived from dA/drho
        requires_vle=False,
        target_key="pressure_res_Pa",
        data_column="pressure_res_Pa",
        vle_init_columns=[],
        output_index=2,
        min_weight_floor=0.5,
        ema_loss=0.1,
    ),
    "chemical_potential_res_J_mol": GoalMetadata(
        model_index=0,
        model_name="HEOS_forward_train",
        target_normalizer="_extract_grad_rho_from_mu_res",
        target_normalizer_args=[[3],[],[0]],
        requires_gradients=True,  # Derived from A + rho*dA/drho
        requires_vle=False,
        target_key="chemical_potential_res_J_mol",
        data_column="chemical_potential_res_J_mol",
        vle_init_columns=[],
        output_index=3,
        min_weight_floor=0.1,
        ema_loss=0.1,
    ),
    "pressure_sat_Pa_liq": GoalMetadata(
        model_index=1,
        model_name="HEOS_vle_norm",
        target_normalizer="_normalize_p_sat",
        target_normalizer_args=[[0],[],[]],
        requires_gradients=True,  # VLE solver uses gradients internally
        requires_vle=True,
        target_key="pressure_sat_Pa_liq",
        data_column="pressure_sat_Pa",   # CSV column name (unchanged)
        # NOTE: VLE init columns are the de-facto inputs besides T, we can just treat them as a input
        #       ... in the sense of T,rho
        vle_init_columns=["molar_density_vapor_mol_m3", "molar_density_liquid_mol_m3"],
        output_index=2,
        min_weight_floor=0.01,
        ema_loss=0.1,
    ),
    "pressure_sat_Pa_gas": GoalMetadata(
        model_index=1,
        model_name="HEOS_vle_norm",
        target_normalizer="_normalize_p_sat",
        target_normalizer_args=[[0],[],[]],
        requires_gradients=True,
        requires_vle=True,
        target_key="pressure_sat_Pa_gas",
        data_column="pressure_sat_Pa",   # same CSV column as liquid phase
        vle_init_columns=["molar_density_vapor_mol_m3", "molar_density_liquid_mol_m3"],
        output_index=2,                  # grad_rho output; prediction uses returns_vap in forward()
        min_weight_floor=0.01,
        #max_weight=0.5, ## ???
        ema_loss=0.1,
    ),
    "chemical_potential_sat_J_mol": GoalMetadata(
        model_index=1,
        model_name="HEOS_vle_norm",
        target_normalizer="_normalize_mu_sat",
        target_normalizer_args=[[1],[],[]],
        requires_gradients=True,  # Derived from A + rho*dA/drho
        requires_vle=True,
        target_key="chemical_potential_sat_J_mol",
        data_column="chemical_potential_sat_J_mol",
        vle_init_columns=["molar_density_vapor_mol_m3", "molar_density_liquid_mol_m3"],
        output_index=3,
        min_weight_floor=0.01,
        ema_loss=0.1,
    ),
    "molar_vaporization_enthalpy_J_mol": GoalMetadata(
        model_index=1,
        model_name="HEOS_vle_norm",
        target_normalizer="<dh_vap>",           # handled in get_goals()
        target_normalizer_args=[],
        requires_gradients=True,
        requires_vle=True, 
        target_key="molar_vaporization_enthalpy_J_mol",
        data_column="molar_vaporization_enthalpy_J_mol",
        vle_init_columns=["molar_density_vapor_mol_m3", "molar_density_liquid_mol_m3"],
        output_index=1,  
        # output_index=0,                         # trains grad_T (entropy) in VLE model output
        min_weight_floor=0.01,
        ema_loss=0.1,
    ),
    "a_res_from_p_res": GoalMetadata(
        model_index=0,
        model_name="HEOS_forward_train",
        target_normalizer="<a_res_from_p_res>",  # handled in get_goals() or _normalize_a_res ??
        target_normalizer_args=[],
        requires_gradients=True,
        requires_vle=False,
        target_key="a_res_from_p_res",       # virtual routing key
        data_column="pressure_res_Pa",        # actual CSV column
        vle_init_columns=[],
        output_index=0,                       # trains the a_res output
        min_weight_floor=0.02,
        ema_loss=0.1,
    ),
    "log_p_consistency": GoalMetadata(
        model_index=1,
        model_name="HEOS_vle_norm",
        target_normalizer="<self_supervised>",
        target_normalizer_args=[],
        requires_gradients=True,
        requires_vle=True,
        target_key="log_p_consistency",
        data_column=None,          # self-supervised: ln(p_vap/p_liq) → 0 at equilibrium
        vle_init_columns=["molar_density_vapor_mol_m3", "molar_density_liquid_mol_m3"],
        output_index=-1,
        min_weight_floor=0.01,
        max_weight=0.5,            # cap: consistency check shouldn't dominate main VLE goals
        ema_loss=0.1,
    ),
    "ln_fug_equil": GoalMetadata(
        model_index=1,
        model_name="HEOS_vle_norm",
        target_normalizer="<self_supervised>",
        target_normalizer_args=[],
        requires_gradients=True,
        requires_vle=True,
        target_key="ln_fug_equil",
        data_column=None,          # self-supervised: ln_fug_vap == ln_fug_liq at equilibrium
        vle_init_columns=["molar_density_vapor_mol_m3", "molar_density_liquid_mol_m3"],
        output_index=-1,
        min_weight_floor=0.01,
        max_weight=0.5,
        ema_loss=0.1,
    ),
    # ---- transport goals (model_index=2, HEOS_transport) ----
    "viscosity_Pa_s": GoalMetadata(
        model_index=2,
        model_name="HEOS_transport",
        target_normalizer="_normalize_viscosity",
        target_normalizer_args=[],
        requires_gradients=True,
        requires_vle=False,
        target_key="viscosity_Pa_s",
        data_column="viscosity_Pa_s",
        vle_init_columns=[],
        output_index=3,       
        max_weight=10.0,                 # ln_vis index in 5-col forward_transport_train result
        min_weight_floor=0.5,
        ema_loss=0.1,
    ),
    "diffusion_m2_s": GoalMetadata(
        model_index=2,
        model_name="HEOS_transport",
        target_normalizer="_normalize_diffusion",
        target_normalizer_args=[],
        requires_gradients=True,
        requires_vle=False,
        target_key="diffusion_m2_s",
        data_column="diffusion_m2_s",
        vle_init_columns=[],
        output_index=4,                        # ln_diff index in 5-col forward_transport_train result
        max_weight=10.0,       
        min_weight_floor=0.5,
        ema_loss=0.1,
    ),
}

MODEL_REGISTRY: Dict = {
    "HEOS_forward_train": HEOS_forward_train_orchestrator,
    "HEOS_vle_norm":      HEOS_vle_norm_orchestrator,
    "HEOS_transport":     HEOS_transport_orchestrator,
}


class GoalEvaluator:
    """
    Evaluates model predictions for a configurable set of training goals.

    This class handles the complexity of:
    1. Determining which forward pass is needed (basic vs gradient)
    2. Routing to appropriate model methods
    3. Normalizing targets consistently
    4. Computing per-goal losses

    For now, this handles only residual properties (A_RES, S_RES, P_RES, MU_RES).
    VLE support (P_SAT) will be added in Step 4.

    Example:
        evaluator = GoalEvaluator(
            model=model,
            goals={TrainingGoal.A_RES, TrainingGoal.P_RES},
            device=device
        )

        predictions = evaluator.forward_residuals(substance_ids, x_trunk)
        targets_norm = evaluator.normalize_targets(y_target, x_trunk)
        losses = evaluator.compute_losses(predictions, targets_norm)
    """

    def __init__(self,
                 model,  # HEOSNet instance
                 config: Dict,
    ):
        """
        Initialize the evaluator with a model and set of goals.

        """
        self.model = model

        config["target_features"] = list(PRESETS[config["training_preset"]])

        # Auto-register the fugacity equilibrium constraint whenever VLE training is active.
        if any(GOAL_REGISTRY[t].requires_vle for t in config["target_features"]):
            for k in ["ln_fug_equil"]:
                if k not in config["target_features"]:
                    config["target_features"].append(k)

        config["compute_gradients"] = False
        config["compute_gradient2_rhorho"] = False

        model_names = []
        model_out_idxs = []
        model_indexes = []
        
        ema_losses = []
        min_weight_floors = []
        max_weight_ceilings = []

        model_count = 0
        for i,target in enumerate(config["target_features"]):
            if GOAL_REGISTRY[target].requires_gradients:
                config["compute_gradients"] = True
            model_name = GOAL_REGISTRY[target].model_name
            # get model needed
            if i == 0: 
                model_names.append(model_name)
            elif model_name != model_names[-1]:
                model_names.append(model_name)
                model_count += 1
            model_indexes.append(model_count)
            model_out_idxs.append(GOAL_REGISTRY[target].output_index)
                        
            ema_losses.append(GOAL_REGISTRY[target].ema_loss)
            min_weight_floors.append(GOAL_REGISTRY[target].min_weight_floor)
            max_weight_ceilings.append(GOAL_REGISTRY[target].max_weight)

        self.ema_losses = torch.tensor(ema_losses, dtype=torch.float32)
        self.min_weight_floors = torch.tensor(min_weight_floors, dtype=torch.float32)
        self.max_weight_ceilings = torch.tensor(
            [x if x != float('inf') else 1e9 for x in max_weight_ceilings],
            dtype=torch.float32
        )

        # Resolve goal keys to actual CSV columns (virtual goals reuse an existing column)
        data_columns = [GOAL_REGISTRY[key].data_column for key in config["target_features"]]

        p = np.flatnonzero(np.diff(model_indexes))+1
        self.model_call_indexes = np.split(model_indexes, p)
        self.model_output_indexes = np.split(model_out_idxs, p)
        self.model_outputs = np.split(config["target_features"], p)   # goal keys for routing
        p_list = [0] + list(p) + [len(data_columns)]
        self.model_data_columns = [data_columns[p_list[i]:p_list[i+1]] for i in range(len(p_list)-1)]
        self.model_names = model_names

        self.model_orchestrators = []
        self.model_forward = []
        for i,key in enumerate(model_names):
            return_keys = self.model_outputs[i]
            orchestrator = MODEL_REGISTRY[key](model, config, return_keys)
            self.model_orchestrators.append(orchestrator)
            self.model_forward.append(orchestrator.model_forward)

        config["input_features"] = self.model_forward
        
        # Loss function
        self._loss_fn = nn.MSELoss()

        # Build data paths from active orchestrators (one path per orchestrator, in order).
        # Each model_name maps to a config key pair; must match the orchestrator sequence.
        _data_path_keys = {
            "HEOS_forward_train": ("data_csv_train",           "data_csv_val"),
            "HEOS_vle_norm":      ("vle_data_csv_train",       "vle_data_csv_val"),
            "HEOS_transport":     ("transport_data_csv_train", "transport_data_csv_val"),
        }
        data_csv_train_paths = []
        data_csv_val_paths   = []
        for name in model_names:
            train_key, val_key = _data_path_keys[name]
            train_path = config.get(train_key)
            val_path   = config.get(val_key)
            if not train_path or not val_path:
                raise ValueError(
                    f"Orchestrator '{name}' is active (goal in preset) but "
                    f"config keys '{train_key}' / '{val_key}' are missing or null."
                )
            data_csv_train_paths.append(train_path)
            data_csv_val_paths.append(val_path)            
        print(data_csv_val_paths)

        print("model_forward",self.model_forward)
        print("model_outputs",self.model_outputs)
        print("model_data_columns",self.model_data_columns)
        # Strip self-supervised goals (data_column=None) before passing to dataset,
        # so y_targets columns match only the data-backed return_keys.
        model_data_columns_filtered = [
            [c for c in cols if c is not None]
            for cols in self.model_data_columns
        ]
        # create train datasets
        self.data_train = ListedDataset(
            data_csv_paths = data_csv_train_paths,
            id_column = config["id_column"],
            input_features = self.model_forward,
            target_features = model_data_columns_filtered,
        )
        # create val datasets
        self.data_val = ListedDataset(
            data_csv_paths = data_csv_val_paths,
            id_column = config["id_column"],
            input_features = self.model_forward,
            target_features = model_data_columns_filtered,
        )        
        self.initial_losses = None
     
        self.config = config
        return

        
    def create_dataloaders(
        self,
        batch_size: int = 64
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for property training.

        Parameters:
        -----------
        batch_size : int
            Batch size

        Returns:
        --------
        train_loader, val_loader : Tuple[DataLoader, DataLoader]
        """

        train_loader = DataLoader(self.data_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.data_val, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def forward(self, batches, return_jacobian=False, update_ema=True):

        losses = []
        jacobian_norms = []
        for i, batch in enumerate(batches):
            # Split input
            substance_ids = batch[0]
            x_in = batch[1]
            y_targets = batch[2]

            # Call models
            model_orchestrator = self.model_orchestrators[i]
            result = model_orchestrator.forward(
                substance_ids, x_in, y_targets, return_jacobian=return_jacobian
            )
            predictions, goals = result[0], result[1]
            valid_mask = result[2] if len(result) > 2 else None
            if return_jacobian and len(result) > 3:
                jacobian_norms.append(result[3])

            # Compute masked losses: skip samples where y_targets was NaN
            for j, (prediction, goal) in enumerate(zip(predictions.T, goals.T)):
                if valid_mask is not None:
                    valid = valid_mask[:, j] & torch.isfinite(goal)
                    if valid.any():
                        loss = self._loss_fn(prediction[valid], goal[valid])
                    else:
                        # No data in this batch for this goal — hold EMA steady
                        loss_idx = len(losses)
                        loss = self.ema_losses[loss_idx].detach() if loss_idx < len(self.ema_losses) else torch.tensor(0.0, device=prediction.device)
                else:
                    loss = self._loss_fn(prediction, goal)
                losses.append(loss)

        # Aggregate losses
        losses = torch.stack(losses)

        # weight losses
        dummy = compute_adaptive_weights_normalized(
            losses, self.ema_losses, self.min_weight_floors, self.initial_losses,
            max_weight_ceilings=self.max_weight_ceilings)
        weights, ema_losses, initial_losses = dummy
        if update_ema:
            self.ema_losses = ema_losses.detach()
            self.initial_losses = initial_losses.detach()

        if return_jacobian and jacobian_norms:
            return losses, weights, torch.cat(jacobian_norms)
        return losses, weights

# =============================================================================
