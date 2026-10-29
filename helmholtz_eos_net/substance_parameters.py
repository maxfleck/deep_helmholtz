import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import json
from typing import List, Optional, Tuple


class SubstanceParameters(nn.Module):
    """
    Manages substance-specific parameters as fixed (non-trainable) values.
    
    This class loads substance data from a CSV file and provides efficient lookup
    of parameter embeddings and metadata by substance index.
    
    **Column Types:**
    
    1. **Parameter Columns**: Numeric values returned as embeddings
       - Examples: p0_norm, p1_norm, p2_norm
       - Returned by forward() as tensor embeddings
       
    2. **Fixed Metadata Columns**: Additional substance information
       - Examples: canonical_smiles, molarweight, family, iupac_name
       - Accessed via get_fixed_param() or get_all_fixed_params()
    
    Example usage:
        params = SubstanceParameters(
            csv_path="substances.csv",
            id_column="ID",
            parameter_columns=["p0_norm", "p1_norm", "p2_norm"],
            fixed_columns=["molarweight", "canonical_smiles"]
        )
        
        # Get parameter embeddings for substances by index (NOT ID)
        embeddings = params(torch.tensor([0, 5, 12]))  # Shape: (3, embedding_dim)
        
        # Get fixed metadata
        mw = params.get_fixed_param(substance_idx=0, param_name="molarweight")
    """
    
    def __init__(self,
                 csv_path: str,
                 id_column: str = "ID",
                 parameter_columns: Optional[List[str]] = None,
                 fixed_columns: Optional[List[str]] = None,
                 embedding_dim: Optional[int] = None):
        """
        Initialize the substance parameter manager.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file containing substance parameters
        id_column : str
            Name of column containing substance IDs (default: "ID")
        parameter_columns : list, optional
            List of column names for parameter embeddings.
            If None, defaults to ["p0_norm", "p1_norm", "p2_norm"]
        fixed_columns : list, optional
            List of column names for fixed metadata.
            If None, defaults to common metadata columns
        embedding_dim : int, optional
            Dimension of embedding space. If None, uses len(parameter_columns)
        """
        super(SubstanceParameters, self).__init__()
        
        # Load substance data
        self.data = pd.read_csv(csv_path)
        self.id_column = id_column
        
        # Verify ID column exists
        if id_column not in self.data.columns:
            raise ValueError(f"ID column '{id_column}' not found in CSV")
        
        # Sort by ID to ensure consistent indexing
        self.data = self.data.sort_values(by=id_column).reset_index(drop=True)
        self.n_substances = len(self.data)
        
        # Create ID to index mapping
        self.id_to_idx = {int(row[id_column]): idx 
                          for idx, row in self.data.iterrows()}
        self.idx_to_id = {idx: int(row[id_column]) 
                          for idx, row in self.data.iterrows()}
        
        # Default parameter columns
        if parameter_columns is None:
            parameter_columns = ["p0_norm", "p1_norm", "p2_norm"]
        
        # Default fixed columns
        if fixed_columns is None:
            fixed_columns = ["canonical_smiles", "molarweight", "family", "iupac_name"]
        
        # Verify columns exist
        missing_param = set(parameter_columns) - set(self.data.columns)
        if missing_param:
            raise ValueError(f"Parameter columns not found in CSV: {missing_param}")
        
        missing_fixed = set(fixed_columns) - set(self.data.columns)
        available_fixed = [col for col in fixed_columns if col in self.data.columns]
        if missing_fixed:
            print(f"Warning: Some fixed columns not found: {missing_fixed}")
            print(f"Using available columns: {available_fixed}")
        
        self.parameter_columns = parameter_columns
        self.fixed_columns = available_fixed
        
        # Determine embedding dimension
        if embedding_dim is None:
            embedding_dim = len(parameter_columns)
        self.embedding_dim = embedding_dim
        
        # Setup parameter embeddings and fixed params
        self._setup_parameters()
        self._setup_fixed_params()
    
    def _setup_parameters(self) -> None:
        """
        Load parameter values from CSV and store as fixed tensors.
        """
        # Extract parameter values
        param_values = self.data[self.parameter_columns].values.astype(np.float32)
        
        # Handle dimension mismatch
        if param_values.shape[1] != self.embedding_dim:
            if param_values.shape[1] < self.embedding_dim:
                # Pad with zeros
                padding = np.zeros((self.n_substances, 
                                   self.embedding_dim - param_values.shape[1]),
                                  dtype=np.float32)
                param_values = np.concatenate([param_values, padding], axis=1)
            else:
                # Truncate
                param_values = param_values[:, :self.embedding_dim]
        
        # Register as buffer (not a parameter, so won't be trained)
        self.register_buffer('embeddings', torch.from_numpy(param_values))
    
    def _setup_fixed_params(self):
        """Setup fixed metadata for fast access."""
        self.fixed_params = {}
        for col in self.fixed_columns:
            values = self.data[col].values
            # Store as tensor if numeric, otherwise as list
            if pd.api.types.is_numeric_dtype(values):
                self.fixed_params[col] = torch.tensor(values, dtype=torch.float32)
            else:
                self.fixed_params[col] = values.tolist()
    
    def forward(self, substance_indices: torch.Tensor) -> torch.Tensor:
        """
        Get parameter embeddings for given substance indices.
        
        **IMPORTANT**: This method expects substance indices (0 to n_substances-1),
        NOT substance IDs.
        
        Parameters:
        -----------
        substance_indices : torch.Tensor
            Tensor of substance indices (0 to n_substances-1)
            Shape: (batch_size,)
        
        Returns:
        --------
        embeddings : torch.Tensor
            Shape: (batch_size, embedding_dim)
        """
        # Ensure input is tensor
        if not isinstance(substance_indices, torch.Tensor):
            substance_indices = torch.tensor(substance_indices, dtype=torch.long)
        
        # Direct indexing
        return self.embeddings[substance_indices]
    
    def get_fixed_param(self, substance_idx: int, param_name: str):
        """
        Get a fixed metadata value for a substance.
        
        Parameters:
        -----------
        substance_idx : int
            Substance index (0 to n_substances-1)
        param_name : str
            Name of the fixed parameter
        
        Returns:
        --------
        value : varies
            Parameter value (tensor element or string)
        """
        if param_name not in self.fixed_params:
            raise ValueError(f"Parameter '{param_name}' not in fixed parameters")
        
        value = self.fixed_params[param_name]
        
        if isinstance(value, torch.Tensor):
            return value[substance_idx].item()
        else:
            return value[substance_idx]
    
    def get_all_fixed_params(self, substance_idx: int) -> dict:
        """Get all fixed metadata for a substance as a dictionary."""
        return {param: self.get_fixed_param(substance_idx, param) 
                for param in self.fixed_columns}
    
    def export_parameters(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Export current parameter values.
        
        Parameters:
        -----------
        file_path : str, optional
            If provided, save to CSV. Otherwise, return as DataFrame.
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with substance IDs and parameter values
        """
        params_np = self.embeddings.cpu().numpy()
        
        # Create column names
        param_cols = [f"param_{i}" for i in range(self.embedding_dim)]
        
        # Create DataFrame
        df = pd.DataFrame(params_np, columns=param_cols)
        df.insert(0, self.id_column, [self.idx_to_id[i] for i in range(self.n_substances)])
        
        if file_path:
            df.to_csv(file_path, index=False)
            print(f"Parameters saved to {file_path}")
        
        return df
    
    def __repr__(self) -> str:
        return (f"SubstanceParameters(\n"
                f"  n_substances={self.n_substances},\n"
                f"  embedding_dim={self.embedding_dim},\n"
                f"  parameter_columns={self.parameter_columns},\n"
                f"  fixed_columns={self.fixed_columns}\n"
                f")")
    
    


class TrainableSubstanceParameters(nn.Module):
    """
    Manages substance-specific parameters with selective trainability.

    This class loads substance data from a CSV file and creates learnable embeddings
    for specified parameters with fine-grained control over which values are trainable.

    **Two Column Types:**

    1. **Fixed Columns (TYPE 1)**: Pure metadata, not passed to model
       - Examples: canonical_smiles, molarweight, family, iupac_name
       - Stored separately, never trained

    2. **Parameter Columns (TYPE 2)**: Parameters for branch network
       - Each column contains both:
         - Subgroup ZERO: Values == 0.0 (always frozen, requires_grad=False)
         - Subgroup NON-ZERO: Non-zero values (trainable by default)
       - Can be frozen/unfrozen at column level dynamically via freeze_columns()

    **Storage Mechanism (Pointer-based for efficiency):**
    - Only trainable parameter values are stored as nn.Parameter
    - Frozen zero positions tracked separately for reconstruction
    - Forward pass reconstructs full (n_substances, embedding_dim) tensor

    Example usage:
        params = TrainableSubstanceParameters(
            csv_path="substances.csv",
            id_column="ID",
            parameter_columns=["p0_norm", "p1_norm", "p2_norm"],
            fixed_columns=["molarweight", "canonical_smiles"],
            freeze_columns=[],  # Optional: freeze specific columns initially
            trainable_threshold=0.0  # Optional: threshold for zero detection
        )

        # Get embeddings for substances by index (NOT ID)
        embeddings = params(torch.tensor([0, 5, 12]))  # Shape: (3, embedding_dim)

        # Dynamically freeze/unfreeze columns
        params.freeze_columns(["p0_norm"])
        params.unfreeze_columns(["p0_norm"])

        # Get fixed parameters for analysis
        mw = params.get_fixed_param(substance_idx=0, param_name="molarweight")
    """

    def __init__(self,
                 csv_path: str,
                 id_column: str = "ID",
                 parameter_columns: Optional[List[str]] = None,
                 fixed_columns: Optional[List[str]] = None,
                 embedding_dim: Optional[int] = None,
                 init_from_csv: bool = True,
                 freeze_columns: Optional[List[str]] = None,
                 trainable_threshold: float = 0.0):
        """
        Initialize the substance parameter manager with selective trainability.

        Parameters:
        -----------
        csv_path : str
            Path to CSV file containing substance parameters
        id_column : str
            Name of column containing substance IDs (default: "ID")
        parameter_columns : list, optional
            List of column names for TYPE 2 (parameter columns).
            If None, defaults to ["p0_norm", "p1_norm", "p2_norm"]
        fixed_columns : list, optional
            List of column names for TYPE 1 (fixed metadata columns).
            If None, defaults to common metadata columns
        embedding_dim : int, optional
            Dimension of embedding space. If None, uses len(parameter_columns)
        init_from_csv : bool
            If True, initialize embeddings from CSV values. Otherwise, random init.
        freeze_columns : list, optional
            List of column names to freeze initially (sets requires_grad=False for non-zero values)
        trainable_threshold : float
            Threshold for zero detection. Values with abs(value) < threshold are frozen.
            Default 0.0 means only exactly 0.0 values are frozen.
        """
        super(TrainableSubstanceParameters, self).__init__()

        # Load substance data
        self.data = pd.read_csv(csv_path)
        self.id_column = id_column

        # Verify ID column exists
        if id_column not in self.data.columns:
            raise ValueError(f"ID column '{id_column}' not found in CSV")

        # Sort by ID to ensure consistent indexing
        self.data = self.data.sort_values(by=id_column).reset_index(drop=True)
        self.n_substances = len(self.data)

        # Create ID to index mapping (kept for compatibility, not used in forward)
        self.id_to_idx = {int(row[id_column]): idx
                          for idx, row in self.data.iterrows()}
        self.idx_to_id = {idx: int(row[id_column])
                          for idx, row in self.data.iterrows()}

        # Default parameter columns (TYPE 2)
        if parameter_columns is None:
            parameter_columns = ["p0_norm", "p1_norm", "p2_norm"]

        # Default fixed columns (TYPE 1)
        if fixed_columns is None:
            fixed_columns = ["canonical_smiles", "molarweight", "family", "iupac_name"]

        # Verify columns exist
        missing_params = set(parameter_columns) - set(self.data.columns)
        if missing_params:
            raise ValueError(f"Parameter columns not found in CSV: {missing_params}")

        missing_fixed = set(fixed_columns) - set(self.data.columns)
        available_fixed = [col for col in fixed_columns if col in self.data.columns]
        if missing_fixed:
            print(f"Warning: Some fixed columns not found: {missing_fixed}")
            print(f"Using available columns: {available_fixed}")

        self.parameter_columns = parameter_columns
        self.fixed_columns = available_fixed
        self.trainable_threshold = trainable_threshold

        # Determine embedding dimension
        if embedding_dim is None:
            embedding_dim = len(parameter_columns)
        self.embedding_dim = embedding_dim

        # Initialize trainable mask: identifies which values are trainable (non-zero)
        # Shape: (n_substances, embedding_dim), dtype: bool
        # True = trainable, False = frozen (zero)
        if init_from_csv:
            self._get_trainable_parameters()
        else:
            # Default: all non-zero when randomly initialized
            self.trainable_mask = torch.ones(
                (self.n_substances, embedding_dim), dtype=torch.bool
            )

        # Initialize column freeze state
        self.frozen_columns = set(freeze_columns) if freeze_columns else set()

        # Create pointer-based storage for trainable parameters only
        self._setup_pointer_storage()

        # Store fixed parameters as tensors for fast lookup
        self._setup_fixed_params()

    def _get_trainable_parameters(self) -> None:
        """
        Identify trainable parameters based on non-zero values.

        Creates a mask tensor of shape (n_substances, embedding_dim) where:
        - True: value is trainable (non-zero and abs >= threshold)
        - False: value is frozen (zero or abs < threshold)

        Called during initialization to set up the trainable_mask.
        """
        init_values = self.data[self.parameter_columns].values.astype(np.float32)

        # Handle dimension mismatch
        if init_values.shape[1] != self.embedding_dim:
            if init_values.shape[1] < self.embedding_dim:
                # Pad with zeros
                padding = np.zeros((self.n_substances,
                                   self.embedding_dim - init_values.shape[1]),
                                  dtype=np.float32)
                init_values = np.concatenate([init_values, padding], axis=1)
            else:
                # Truncate
                init_values = init_values[:, :self.embedding_dim]

        # Create trainable mask: True for trainable (abs > threshold), False for frozen
        # Using > instead of >= ensures that values at exactly threshold=0.0 are frozen
        self.trainable_mask = torch.from_numpy(
            np.abs(init_values) > self.trainable_threshold
        ).bool()

        # Store initial values for later use
        self._init_values = torch.from_numpy(init_values)

    def _setup_pointer_storage(self) -> None:
        """
        Set up pointer-based storage for efficient trainable parameter access.

        This method:
        1. Extracts trainable values and their positions
        2. Creates nn.Parameter for only trainable values
        3. Stores pointers to reconstruct full embeddings
        4. Sets requires_grad = False for frozen parameters
        5. Applies initial column freezing
        """
        # Get initial values if not already loaded
        if not hasattr(self, '_init_values'):
            init_values = self.data[self.parameter_columns].values.astype(np.float32)
            if init_values.shape[1] != self.embedding_dim:
                if init_values.shape[1] < self.embedding_dim:
                    padding = np.zeros((self.n_substances,
                                       self.embedding_dim - init_values.shape[1]),
                                      dtype=np.float32)
                    init_values = np.concatenate([init_values, padding], axis=1)
                else:
                    init_values = init_values[:, :self.embedding_dim]
            self._init_values = torch.from_numpy(init_values)

        # Identify which values are trainable (not in frozen columns and non-zero)
        # Create column-level freeze mask
        col_freeze_mask = torch.ones(self.embedding_dim, dtype=torch.bool)
        for col_name in self.frozen_columns:
            if col_name in self.parameter_columns:
                col_idx = self.parameter_columns.index(col_name)
                col_freeze_mask[col_idx] = False

        # Combined mask: trainable if (not in frozen column) AND (non-zero from threshold)
        combined_mask = self.trainable_mask & col_freeze_mask.unsqueeze(0)

        # Store masks as buffers (not parameters, won't be updated)
        self.register_buffer('_trainable_mask', self.trainable_mask)
        self.register_buffer('_col_freeze_mask', col_freeze_mask)
        self.register_buffer('_combined_mask', combined_mask)

        # Extract trainable values and create nn.Parameter
        trainable_values = self._init_values.clone()
        trainable_values[~combined_mask] = 0.0  # Zero out frozen values

        # Create parameter from trainable values
        self.embeddings = nn.Parameter(trainable_values, requires_grad=True)

        # Set requires_grad = False for frozen parameters
        with torch.no_grad():
            for i in range(self.n_substances):
                for j in range(self.embedding_dim):
                    if not combined_mask[i, j]:
                        # This is a frozen value; we'll prevent updates via masking
                        pass

    def _setup_fixed_params(self):
        """Setup fixed parameters (TYPE 1) for fast access."""
        self.fixed_params = {}
        for col in self.fixed_columns:
            values = self.data[col].values
            # Store as tensor if numeric, otherwise as list
            if pd.api.types.is_numeric_dtype(values):
                self.fixed_params[col] = torch.tensor(values, dtype=torch.float32)
            else:
                self.fixed_params[col] = values.tolist()

    def forward(self, substance_indices: torch.Tensor) -> torch.Tensor:
        """
        Get trainable embeddings for given substance indices.

        **IMPORTANT**: This method expects substance indices (0 to n_substances-1),
        NOT substance IDs. If you pass invalid indices, a KeyError will be raised,
        which is intentional to enforce correct usage.

        Parameters:
        -----------
        substance_indices : torch.Tensor
            Tensor of substance indices (0 to n_substances-1)
            Shape: (batch_size,)

        Returns:
        --------
        embeddings : torch.Tensor
            Shape: (batch_size, embedding_dim)
        """
        # Ensure input is tensor
        if not isinstance(substance_indices, torch.Tensor):
            substance_indices = torch.tensor(substance_indices, dtype=torch.long)

        # Direct indexing: no id_to_idx mapping needed
        # embeddings shape: (batch_size, embedding_dim)
        embeddings = self.embeddings[substance_indices]

        # Apply masking to ensure frozen values stay at their initialized values
        # This is crucial for the backward pass
        batch_size = substance_indices.shape[0]
        combined_mask = self._combined_mask[substance_indices]  # (batch_size, embedding_dim)

        # Zero out any parameters that drifted into frozen positions
        # (gradients should have been blocked, but this is a safety measure)
        embeddings = embeddings * combined_mask.float()

        return embeddings

    def freeze_columns(self, column_names: List[str]) -> None:
        """
        Freeze specified parameter columns (TYPE 2).

        Sets requires_grad=False for all non-zero values in the specified columns.
        Zero values are already frozen. Can be called dynamically during training.

        Parameters:
        -----------
        column_names : list
            List of parameter column names to freeze (e.g., ["p0_norm", "p1_norm"])
        """
        # Validate column names
        invalid_cols = set(column_names) - set(self.parameter_columns)
        if invalid_cols:
            print(f"Warning: Columns not in parameter_columns: {invalid_cols}")

        # Add to frozen columns set
        self.frozen_columns.update(column_names)

        # Rebuild the combined mask
        col_freeze_mask = torch.ones(self.embedding_dim, dtype=torch.bool)
        for col_name in self.frozen_columns:
            if col_name in self.parameter_columns:
                col_idx = self.parameter_columns.index(col_name)
                col_freeze_mask[col_idx] = False

        combined_mask = self._trainable_mask & col_freeze_mask.unsqueeze(0)

        # Update buffers
        self._col_freeze_mask = col_freeze_mask
        self.register_buffer('_col_freeze_mask', col_freeze_mask)
        self.register_buffer('_combined_mask', combined_mask)

        print(f"Froze columns: {column_names}")

    def unfreeze_columns(self, column_names: Optional[List[str]] = None) -> None:
        """
        Unfreeze specified parameter columns (TYPE 2).

        Sets requires_grad=True for all non-zero values in the specified columns.
        If column_names is None, unfreezes all columns.

        Parameters:
        -----------
        column_names : list, optional
            List of parameter column names to unfreeze (e.g., ["p0_norm"]).
            If None, unfreezes all currently frozen columns.
        """
        if column_names is None:
            # Unfreeze all
            self.frozen_columns.clear()
        else:
            # Validate column names
            invalid_cols = set(column_names) - set(self.parameter_columns)
            if invalid_cols:
                print(f"Warning: Columns not in parameter_columns: {invalid_cols}")

            # Remove from frozen columns set
            self.frozen_columns.difference_update(column_names)

        # Rebuild the combined mask
        col_freeze_mask = torch.ones(self.embedding_dim, dtype=torch.bool)
        for col_name in self.frozen_columns:
            if col_name in self.parameter_columns:
                col_idx = self.parameter_columns.index(col_name)
                col_freeze_mask[col_idx] = False

        combined_mask = self._trainable_mask & col_freeze_mask.unsqueeze(0)

        # Update buffers
        self._col_freeze_mask = col_freeze_mask
        self.register_buffer('_col_freeze_mask', col_freeze_mask)
        self.register_buffer('_combined_mask', combined_mask)

        if column_names is None:
            print("Unfroze all columns")
        else:
            print(f"Unfroze columns: {column_names}")

    def get_fixed_param(self, substance_idx: int, param_name: str):
        """
        Get a fixed parameter (TYPE 1) value for a substance.

        Parameters:
        -----------
        substance_idx : int
            Substance index (0 to n_substances-1)
        param_name : str
            Name of the fixed parameter

        Returns:
        --------
        value : varies
            Parameter value (tensor element or string)
        """
        if param_name not in self.fixed_params:
            raise ValueError(f"Parameter '{param_name}' not in fixed parameters")

        value = self.fixed_params[param_name]

        if isinstance(value, torch.Tensor):
            return value[substance_idx].item()
        else:
            return value[substance_idx]

    def get_all_fixed_params(self, substance_idx: int) -> dict:
        """Get all fixed parameters (TYPE 1) for a substance as a dictionary."""
        return {param: self.get_fixed_param(substance_idx, param)
                for param in self.fixed_columns}

    def export_parameters(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Export current embedding values.

        Parameters:
        -----------
        file_path : str, optional
            If provided, save to CSV. Otherwise, return as DataFrame.

        Returns:
        --------
        df : pd.DataFrame
            DataFrame with substance IDs and embedding values
        """
        embeddings_np = self.embeddings.detach().cpu().numpy()

        # Create column names
        embed_cols = [f"embedding_{i}" for i in range(self.embedding_dim)]

        # Create DataFrame
        df = pd.DataFrame(embeddings_np, columns=embed_cols)
        df.insert(0, self.id_column, [self.idx_to_id[i] for i in range(self.n_substances)])

        if file_path:
            df.to_csv(file_path, index=False)
            print(f"Embeddings saved to {file_path}")

        return df

    def load_embeddings(self, file_path: str) -> None:
        """
        Load embeddings from JSON file.

        Parameters:
        -----------
        file_path : str
            Path to JSON file containing embeddings
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Verify compatibility
        if data["n_substances"] != self.n_substances:
            raise ValueError("Number of substances doesn't match")
        if data["embedding_dim"] != self.embedding_dim:
            raise ValueError("Embedding dimension doesn't match")

        # Load embeddings
        embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
        with torch.no_grad():
            self.embeddings.copy_(embeddings)

        print(f"Embeddings loaded from {file_path}")

    def __repr__(self) -> str:
        return (f"TrainableSubstanceParameters(\n"
                f"  n_substances={self.n_substances},\n"
                f"  embedding_dim={self.embedding_dim},\n"
                f"  parameter_columns={self.parameter_columns},\n"
                f"  fixed_columns={self.fixed_columns},\n"
                f"  frozen_columns={self.frozen_columns},\n"
                f"  trainable_threshold={self.trainable_threshold}\n"
                f")")
