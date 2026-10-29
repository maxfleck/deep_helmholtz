from si_units import *  # SI units package (separate from feos in v0.8.0)
from feos.pcsaft import PcSaftParameters, PcSaftRecord, Identifier, PureRecord
from feos.eos import EquationOfState, State, PhaseDiagram
from feos.eos import Contributions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def load_saft_parameters(smiles, saft_csv_path):
    """
    Load PC-SAFT parameters for a given SMILES from a CSV file.
    
    Parameters
    ----------
    smiles : str
        Canonical SMILES string of the molecule
    saft_csv_path : str
        Path to the tab-delimited CSV file with PC-SAFT parameters
    
    Returns
    -------
    parameters : PcSaftParameters
        PC-SAFT parameters object
    eos : EquationOfState
        Equation of state object
    """
    # Read the CSV file (tab-delimited)
    # saft_df = pd.read_csv(saft_csv_path, delimiter='\t')
    saft_df = pd.read_csv(saft_csv_path)
    
    # Find the row matching the SMILES
    species_rows = saft_df[saft_df['canonical_smiles'] == smiles]
    
    if len(species_rows) == 0:
        raise ValueError(f"SMILES '{smiles}' not found in {saft_csv_path}")
    
    # Use the first match
    row = species_rows.iloc[0]
    
    # Create identifier
    identifier = Identifier(
        name=row['common_name'],
        smiles=row['canonical_smiles'],
        cas=row['cas']
    )
    
    # Create PC-SAFT record with parameters
    # Handle optional association parameters
    record = PcSaftRecord(
        m=row['m'],                    # number of segments
        sigma=row['sigma'],            # segment diameter [Å]
        epsilon_k=row['epsilon_k'],    # energy parameter [K]
        mu=row['mu'] if pd.notna(row['mu']) else None,  # dipole moment [D]
        kappa_ab=row['kappa_ab'] if pd.notna(row['kappa_ab']) else None,  # association volume
        epsilon_k_ab=row['epsilon_k_ab'] if pd.notna(row['epsilon_k_ab']) else None,  # association energy [K]
        na=int(row['na']) if pd.notna(row['na']) else None,  # number of acceptor sites
        nb=int(row['nb']) if pd.notna(row['nb']) else None   # number of donor sites
    )
    
    # Create pure record
    pure_record = PureRecord(
        identifier=identifier,
        molarweight=row['molarweight'], #* GRAM / MOL,  # DONT Convert to SI units
        model_record=record
    )
    
    # Create parameters and EoS
    parameters = PcSaftParameters.from_records([pure_record])
    eos = EquationOfState.pcsaft(parameters)
    
    return parameters, eos


def calculate_vle(eos, T_min_factor=0.5, npoints=501):
    """
    Calculate vapor-liquid equilibrium (VLE) and critical point.
    
    Parameters
    ----------
    eos : EquationOfState
        Equation of state object
    T_min_factor : float, optional
        Factor to multiply with T_c to get minimum temperature (default: 0.5)
    npoints : int, optional
        Number of points in VLE curve (default: 501)
    
    Returns
    -------
    vle_df : pd.DataFrame
        DataFrame with VLE properties (temperature, pressure, densities)
    critical_point : State
        Critical point state
    """
    # Calculate critical point
    critical_point = State.critical_point(eos)
    T_c = critical_point.temperature / KELVIN
    P_c = critical_point.pressure() / PASCAL
    
    # Calculate minimum temperature
    T_min = T_min_factor * T_c * KELVIN
    
    # Calculate VLE phase diagram
    phase_diagram = PhaseDiagram.pure(eos, min_temperature=T_min, npoints=npoints)
    
    # Extract VLE data
    temperatures = phase_diagram.vapor.temperature / KELVIN
    pressures = phase_diagram.vapor.pressure / PASCAL
    density_liquid = phase_diagram.liquid.density / (MOL / METER**3)
    density_vapor = phase_diagram.vapor.density / (MOL / METER**3)
    
    # Create DataFrame
    vle_df = pd.DataFrame({
        'temperature_K': temperatures,
        'pressure_sat_Pa': pressures,
        'molar_density_liquid_mol_m3': density_liquid,
        'molar_density_vapor_mol_m3': density_vapor
    })
    
    return vle_df, critical_point


def generate_grid(vle_df, n_isotherms=10, n_isochores=10, T_factor=1.2,
                  rho_min_factor=1, n_low_density_cols=0):
    """
    Generate a (ρ, T) grid based on VLE results.

    Parameters
    ----------
    vle_df : pd.DataFrame
        VLE data with columns: temperature, pressure_sat, density_liquid_vle, density_vapor_vle
    n_isotherms : int, optional
        Number of isotherms (temperature points) (default: 10)
    n_isochores : int, optional
        Number of isochores (density points) (default: 10)
    T_factor : float, optional
        Maximum temperature = T_factor × T_critical (default: 1.2)
    rho_min_factor : float, optional
        Multiplier on min(rho_vapor) that sets the lower bound of the main density grid.
        0 (default) uses min(rho_vapor) directly. Values > 0 scale from that minimum,
        e.g. rho_min_factor=800 gives rho_min ≈ 100 mol/m³ for typical light molecules.
    n_low_density_cols : int, optional
        Extra isochores added below the main grid's rho_min, log-spaced between
        1% and 90% of min(rho_vapor). These are always below the saturation vapor
        density at every isotherm and survive the two-phase filter (default: 0)

    Returns
    -------
    grid_df : pd.DataFrame
        DataFrame with (ρ, T) grid points
    """
    # Get temperature range from VLE
    T_min = vle_df['temperature_K'].min()
    T_max = vle_df['temperature_K'].max()  # This is T_critical
    T_grid_max = T_factor * T_max

    if n_low_density_cols > 2:
        rho_min = vle_df['molar_density_vapor_mol_m3'].mean()
    elif n_low_density_cols == 0:        
        rho_min = vle_df['molar_density_vapor_mol_m3'].min()
    else:
        rho_min = 100 # old old
    rho_max = vle_df['molar_density_liquid_mol_m3'].max() * 1.5
    print("Density range for grid: ", rho_min, rho_max)

    # Generate temperature grid (linear spacing)
    temperatures = np.linspace(T_min, T_grid_max, n_isotherms)

    # Generate density grid (linear spacing)
    densities = np.linspace(rho_min, rho_max, n_isochores)

    # Extra isochores below rho_min, guaranteed below rho_vapor(T) for every isotherm
    if n_low_density_cols > 0:
        #low_densities = np.linspace(
        #    vle_df['molar_density_vapor_mol_m3'].min(),
        #    rho_min*rho_min_factor,
        #    n_low_density_cols
        #)
        print( densities[:10] )
        # without rho_min_factor factor !!
        rho_mins = np.sort(densities).squeeze()[:2]
        low_densities = np.linspace(
            rho_mins.min(),
            rho_mins.max(),
            n_low_density_cols+1
        )[1:-1]
        print("low_densities", low_densities)
        densities = np.concatenate([low_densities, densities])
        densities = np.unique(densities)  # Remove duplicates if low_densities overlap with main grid

    # Create meshgrid (density on x-axis, temperature on y-axis for better visualization)
    rho_mesh, T_mesh = np.meshgrid(densities, temperatures)

    # Flatten to create DataFrame
    grid_df = pd.DataFrame({
        'temperature_K': T_mesh.flatten(),
        'molar_density_mol_m3': rho_mesh.flatten()
    })

    return grid_df


def add_vle_properties(grid_df, vle_df):
    """
    Add interpolated VLE properties to the grid DataFrame.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid with temperature_K and molar_density_mol_m3 columns
    vle_df : pd.DataFrame
        VLE data

    Returns
    -------
    grid_df : pd.DataFrame
        Grid with added VLE properties
    """
    # Create interpolation functions
    T_vle = vle_df['temperature_K'].values

    interp_P_sat = interp1d(T_vle, vle_df['pressure_sat_Pa'].values,
                             kind='linear', fill_value='extrapolate')
    interp_rho_liq = interp1d(T_vle, vle_df['molar_density_liquid_mol_m3'].values,
                               kind='linear', fill_value='extrapolate')
    interp_rho_vap = interp1d(T_vle, vle_df['molar_density_vapor_mol_m3'].values,
                               kind='linear', fill_value='extrapolate')

    # Interpolate VLE properties for each grid point
    grid_df['pressure_sat_Pa'] = interp_P_sat(grid_df['temperature_K'])
    grid_df['molar_density_liquid_mol_m3'] = interp_rho_liq(grid_df['temperature_K'])
    grid_df['molar_density_vapor_mol_m3'] = interp_rho_vap(grid_df['temperature_K'])

    return grid_df


def calculate_state_point(row, eos, P_min, P_max, T_c=None):
    """
    Calculate state properties for a single (T, ρ) point.

    This function is designed to be used with df.apply().

    Parameters
    ----------
    row : pd.Series
        Row from DataFrame with temperature_K, molar_density_mol_m3, and VLE properties
    eos : EquationOfState
        Equation of state object
    P_min : float
        Minimum pressure from VLE (Pascal)
    P_max : float
        Maximum pressure from VLE (Pascal)
    T_c : float, optional
        Critical temperature (K). If given, P <= 0 points inside the two-phase region
        (T < T_c, ρ_vap < ρ < ρ_liq) are kept with phase 'two_phase'; P <= 0 outside it
        is still rejected. If None, all P <= 0 points are rejected (default: None)

    Returns
    -------
    result : pd.Series
        Series with calculated properties or 'fail' phase if outside pressure window
    """
    T = row['temperature_K'] * KELVIN
    rho = row['molar_density_mol_m3']  # mol/m^3
    P_sat = row['pressure_sat_Pa'] * PASCAL

    try:
        # Convert density to volume
        # rho [mol/m^3], n_moles = 1 mol
        # V = n / rho = 1 mol / rho [mol/m^3] = m^3
        n_moles = 1.0 * MOL
        volume = n_moles / (rho * MOL / METER**3)

        # Create state at given T and V (N is fixed)
        state = State(eos, temperature=T, volume=volume, total_moles=n_moles)

        # Extract pressure and properties
        P = state.pressure()
        P_res = P / PASCAL - 8.314*T*rho / KELVIN
        #print(P_res)

        in_two_phase = (T_c is not None and row['temperature_K'] < T_c and
                        row['molar_density_vapor_mol_m3'] < rho < row['molar_density_liquid_mol_m3'])
        negative_two_phase = in_two_phase and P <= 0 * PASCAL

        # Check if pressure is within valid window (P <= 0 also rejects Z <= 0,
        # except inside the two-phase region when T_c is given)
        if  P > P_max * PASCAL or (P <= 0 * PASCAL and not negative_two_phase):
            return pd.Series({
                'phase': 'fail',
                'molar_density_mol_m3': rho,
                'pressure_Pa': P / PASCAL,
                'pressure_res_Pa': P_res,
                'chemical_potential_res_J_mol': np.nan,
                'entropy_res_J_K': np.nan,
                'helmholtz_energy_res_J_mol': np.nan,
                'pressure_sat_Pa': row['pressure_sat_Pa'],
                'molar_density_liquid_vle_mol_m3': row['molar_density_liquid_mol_m3'],
                'molar_density_vapor_vle_mol_m3': row['molar_density_vapor_mol_m3'],
            })

        # Extract molar density (verify)
        molar_density = state.density / (MOL / METER**3)
        chemical_potential_res = state.chemical_potential(Contributions.Residual)[0] / (JOULE / MOL)
        helmholtz_energy_res = state.molar_helmholtz_energy(Contributions.Residual) / (JOULE / MOL)
        entropy_res = state.entropy(Contributions.Residual) / (JOULE / KELVIN)

        # Determine phase based on pressure relative to saturation
        if negative_two_phase:
            phase = 'two_phase'
        elif P < P_sat:
            phase = 'vapor'
        else:
            phase = 'liquid'

        #print(molar_density, rho)
        grad_rho_p = P_res / (molar_density **2)
        grad_rho_mu = (chemical_potential_res - helmholtz_energy_res ) / (molar_density )
        #print(grad_rho_p, grad_rho_mu)

        # Check if validity
        if np.abs(grad_rho_p- grad_rho_mu ) > .005:
            return pd.Series({
                'phase': 'fail',
                'molar_density_mol_m3': rho,
                'pressure_Pa': P / PASCAL,
                'pressure_res_Pa': P_res,
                'chemical_potential_res_J_mol': np.nan,
                'entropy_res_J_K': np.nan,
                'helmholtz_energy_res_J_mol': np.nan,
                'pressure_sat_Pa': row['pressure_sat_Pa'],
                'molar_density_liquid_vle_mol_m3': row['molar_density_liquid_mol_m3'],
                'molar_density_vapor_vle_mol_m3': row['molar_density_vapor_mol_m3'],
            })


        return pd.Series({
            'phase': phase,
            'molar_density_mol_m3': molar_density,
            'pressure_Pa': P / PASCAL,
            'pressure_res_Pa': P_res,
            'chemical_potential_res_J_mol': chemical_potential_res,
            'entropy_res_J_K': entropy_res,
            'helmholtz_energy_res_J_mol': helmholtz_energy_res,
            'pressure_sat_Pa': row['pressure_sat_Pa'],
            'molar_density_liquid_vle_mol_m3': row['molar_density_liquid_mol_m3'],
            'molar_density_vapor_vle_mol_m3': row['molar_density_vapor_mol_m3'],
        })

    except Exception as e:
        # If calculation fails, mark as failed
        return pd.Series({
            'phase': 'fail',
            'molar_density_mol_m3': rho,
            'pressure_Pa': np.nan,
            'pressure_res_Pa': np.nan,
            'chemical_potential_res_J_mol': np.nan,
            'entropy_res_J_K': np.nan,
            'helmholtz_energy_res_J_mol': np.nan,
            'pressure_sat_Pa': row['pressure_sat_Pa'],
            'molar_density_liquid_vle_mol_m3': row['molar_density_liquid_mol_m3'],
            'molar_density_vapor_vle_mol_m3': row['molar_density_vapor_mol_m3'],
        })


def generate_state_points(smiles, saft_csv_path,
                         n_isotherms=10, n_isochores=10, P_min=None,
                         T_factor=1.2, P_factor=1.2, T_min_factor=0.5,
                         filter_two_phase=True, rho_gas_factor=[1.1, 0],
                         rho_liq_factor=[0.9, 0], rho_min_factor=1,
                         n_low_density_cols=0,
                         ideal_gas_rho_norm_threshold=None,
                         keep_negative_two_phase=False,
                         ):
    """
    Generate state points on a (ρ, T) grid for a given species.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string
    saft_csv_path : str
        Path to PC-SAFT parameters CSV file
    n_isotherms : int, optional
        Number of temperature points (default: 10)
    n_isochores : int, optional
        Number of isochore (density) points (default: 10)
    T_factor : float, optional
        Maximum temperature factor (default: 1.2)
    P_factor : float, optional
        Maximum pressure = P_factor × P_max VLE (default: 1.2)
    T_min_factor : float, optional
        Minimum temperature factor for VLE calculation (default: 0.5)
    filter_two_phase : bool, optional
        Filter out two-phase region densities (default: True)
    rho_gas_factor : float, optional
        Lower density cutoff for gas phase = rho_gas_factor × ρ_vapor (default: 1.1)
    rho_liq_factor : float, optional
        Upper density cutoff for liquid phase = rho_liq_factor × ρ_liquid (default: 0.9)
    rho_min_factor : float, optional
        Multiplier on min(rho_vapor) for the lower bound of the main density grid.
        0 (default) uses min(rho_vapor) directly (physically motivated, near ideal gas).
    n_low_density_cols : int, optional
        Extra isochores below rho_min, log-spaced between 1% and 90% of min(rho_vapor).
        Always below the saturation vapor boundary → survive two-phase filter (default: 0)
    keep_negative_two_phase : bool, optional
        Keep P <= 0 points inside the two-phase region (phase 'two_phase'); P <= 0 elsewhere
        is still removed. Combine with filter_two_phase=False to keep the full loop (default: False)

    Returns
    -------
    state_df : pd.DataFrame
        DataFrame with state points and properties
    vle_df : pd.DataFrame
        VLE curve data
    critical_point : State
        Critical point
    """
    print(f"Loading PC-SAFT parameters for {smiles}...")
    parameters, eos = load_saft_parameters(smiles, saft_csv_path)

    print("Calculating VLE and critical point...")
    vle_df, critical_point = calculate_vle(eos, T_min_factor=T_min_factor)

    print(f"Critical point: T_c = {critical_point.temperature/KELVIN:.2f} K, "
          f"P_c = {critical_point.pressure()/PASCAL/1e6:.3f} MPa")

    # Get pressure range from VLE for filtering
    P_min = vle_df['pressure_sat_Pa'].min()
    if P_min is not None:
        P_max = P_factor * vle_df['pressure_sat_Pa'].max()

    print(f"Generating grid with {n_isotherms} isotherms and {n_isochores} isochores...")
    grid_df = generate_grid(vle_df, n_isotherms, n_isochores, T_factor, rho_min_factor, n_low_density_cols)

    print("Adding VLE properties to grid...")
    grid_df = add_vle_properties(grid_df, vle_df)

    # Filter out two-phase region if requested
    if filter_two_phase:
        n_before = len(grid_df)
        # Keep only densities outside the two-phase region:
        # density < -rho_liq_factor + rho_liquid  OR  density > rho_gas_factor + rho_vapor
        rho_gas_factor = np.max([rho_gas_factor[0], rho_gas_factor[1]*np.mean(vle_df['molar_density_vapor_mol_m3'])])
        rho_liq_factor = np.max([rho_liq_factor[0], rho_liq_factor[1]*np.mean(vle_df['molar_density_liquid_mol_m3'])])

        mask = ~((grid_df['molar_density_mol_m3'] > rho_gas_factor + grid_df['molar_density_vapor_mol_m3']) &
                 (grid_df['molar_density_mol_m3'] < -rho_liq_factor + grid_df['molar_density_liquid_mol_m3']))
        grid_df = grid_df[mask].reset_index(drop=True)
        n_filtered = n_before - len(grid_df)
        print(f"Filtered out {n_filtered} points in two-phase region. Remaining: {len(grid_df)} points.")

    print(f"Calculating state points for {len(grid_df)} grid points...")
    # Use apply to calculate state for each row with pressure boundaries
    T_c = critical_point.temperature / KELVIN if keep_negative_two_phase else None
    new_columns = grid_df.apply(lambda row: calculate_state_point(row, eos, P_min, P_max, T_c), axis=1)

    # Convert results to DataFrame columns
    grid_df[new_columns.columns] = new_columns

    if ideal_gas_rho_norm_threshold is not None:
        ig_mask = np.log(grid_df['molar_density_mol_m3']) <= ideal_gas_rho_norm_threshold
        pull_rho = np.exp(ideal_gas_rho_norm_threshold + 0.1)
        # Relocate below-threshold rows to just above the boundary so the model's
        # ideal-gas mask doesn't apply → real gradient signal toward ideal-gas targets.
        grid_df.loc[ig_mask, 'molar_density_mol_m3'] = pull_rho
        grid_df.loc[ig_mask, 'pressure_Pa'] = 8.314 * grid_df.loc[ig_mask, 'temperature_K'] * pull_rho
        grid_df.loc[ig_mask, ['pressure_res_Pa', 'entropy_res_J_K']] = 0.0
        grid_df.loc[ig_mask, 'chemical_potential_res_J_mol'] = -1.0
        grid_df.loc[ig_mask, 'helmholtz_energy_res_J_mol'] = -1.0
        grid_df.loc[ig_mask & (grid_df['phase'] == 'fail'), 'phase'] = 'vapor'
        print(f"Ideal gas pull: relocated {ig_mask.sum()} pts → log(rho)={ideal_gas_rho_norm_threshold + 0.1:.1f}")

    # Filter out failed calculations (outside pressure window or calculation errors)
    n_failed = (grid_df['phase'] == 'fail').sum()
    if n_failed > 0:
        print(f"Warning: {n_failed} state point calculations failed (outside pressure window or errors) and were removed.")

    state_df = grid_df[grid_df['phase'] != 'fail'].reset_index(drop=True)

    print(f"Successfully calculated {len(state_df)} state points.")
    if len(state_df) > 0:
        print(f"Pressure range in results: {state_df['pressure_Pa'].min()/1e6:.2f} - {state_df['pressure_Pa'].max()/1e6:.2f} MPa")

    return state_df, vle_df, critical_point