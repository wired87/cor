from numpy import stack


def calc_psi_bar(psi, gamma_0):
    # Dirac Adjoint: ψ_bar = ψ†γ⁰
    """
    Calc psi bar for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `psi`, `gamma_0`.
    2. Builds intermediate state such as `psi_bar` before applying the main logic.
    3. Delegates side effects or helper work through `psi.conj()`.
    4. Returns the assembled result to the caller.

    Inputs:
    - `psi`: Caller-supplied value used during processing.
    - `gamma_0`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    psi_bar = psi.conj().T @ gamma_0
    return psi_bar

def calc_yukawa_coupling(y, psi_bar, psi, h):
    # Yukawa interaction term: -y * H * (ψ_bar @ ψ)
    """
    Calc yukawa coupling for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `y`, `psi_bar`, `psi`, `h`.
    2. Builds intermediate state such as `yukawa_coupling` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `y`: Caller-supplied value used during processing.
    - `psi_bar`: Caller-supplied value used during processing.
    - `psi`: Caller-supplied value used during processing.
    - `h`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    yukawa_coupling = -y * h * (psi_bar @ psi)
    return yukawa_coupling

def calc_yterm(yukawa_coupling_sum):
    # Total Yukawa contribution collected from neighbor interactions
    """
    Calc yterm for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `yukawa_coupling_sum`.
    2. Builds intermediate state such as `yterm` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `yukawa_coupling_sum`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    yterm = yukawa_coupling_sum
    return yterm

def calc_spatial_diff_psi(psi_, psi__, d):
    # Spatial central difference: (ψ_{i+1} - ψ_{i-1}) / 2d
    """
    Calc spatial diff psi for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `psi_`, `psi__`, `d`.
    2. Builds intermediate state such as `spatial_diff` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `psi_`: Caller-supplied value used during processing.
    - `psi__`: Caller-supplied value used during processing.
    - `d`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    spatial_diff = (psi_ - psi__) / (2.0 * d)
    return spatial_diff

def calc_time_diff_psi(psi, prev_psi, dt):
    # Temporal backward difference: (ψ_t - ψ_{t-dt}) / dt
    """
    Calc time diff psi for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `psi`, `prev_psi`, `dt`.
    2. Builds intermediate state such as `time_diff` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `psi`: Caller-supplied value used during processing.
    - `prev_psi`: Caller-supplied value used during processing.
    - `dt`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    time_diff = (psi - prev_psi) / dt
    return time_diff

def calc_dirac_kinetic_component(gamma_mu, dmu_psi):
    # Kinetic component of Dirac equation: γμ @ ∂μψ
    """
    Calc dirac kinetic component for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `gamma_mu`, `dmu_psi`.
    2. Builds intermediate state such as `dirac_kinetic_component` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `gamma_mu`: Caller-supplied value used during processing.
    - `dmu_psi`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    dirac_kinetic_component = gamma_mu @ dmu_psi
    return dirac_kinetic_component

def calc_mass_term_psi(mass, psi):
    # Fermion mass contribution: -i * m * ψ
    """
    Calc mass term psi for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `mass`, `psi`.
    2. Builds intermediate state such as `mass_term` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `mass`: Caller-supplied value used during processing.
    - `psi`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    mass_term = -1j * mass * psi
    return mass_term

def calc_gterm_mu(i, g, field_value, T, psi):
    # Gauge coupling component: -i * g * Aμ * (T @ ψ)
    """
    Calc gterm mu for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `i`, `g`, `field_value`, `T`.
    2. Builds intermediate state such as `gterm_mu` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `i`: Caller-supplied value used during processing.
    - `g`: Graph instance that the workflow reads from or mutates.
    - `field_value`: Caller-supplied value used during processing.
    - `T`: Caller-supplied value used during processing.
    - `psi`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    gterm_mu = -i * g * field_value * (T @ psi)
    return gterm_mu

def calc_gterm(gterm_mu_sum):
    # Total gauge interaction collected across dimensions and neighbors
    """
    Calc gterm for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `gterm_mu_sum`.
    2. Builds intermediate state such as `gterm` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `gterm_mu_sum`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    gterm = gterm_mu_sum
    return gterm

def calc_dirac(gamma0_inv, dirac_kinetic_sum, gterm, yterm, mass_term):
    # Dirac equation evolution: -γ⁰_inv @ (Σ(γμ∂μψ) + gterm + yterm + mass_term)
    """
    Calc dirac for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `gamma0_inv`, `dirac_kinetic_sum`, `gterm`, `yterm`.
    2. Builds intermediate state such as `dirac` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `gamma0_inv`: Caller-supplied value used during processing.
    - `dirac_kinetic_sum`: Caller-supplied value used during processing.
    - `gterm`: Caller-supplied value used during processing.
    - `yterm`: Caller-supplied value used during processing.
    - `mass_term`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    dirac = -gamma0_inv @ (dirac_kinetic_sum + gterm + yterm + mass_term)
    return dirac

def calc_psi(psi, dt, dirac):
    # First-order Euler update for spinor field: ψ_new = ψ + dt * ∂tψ
    """
    Calc psi for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `psi`, `dt`, `dirac`.
    2. Builds intermediate state such as `psi` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `psi`: Caller-supplied value used during processing.
    - `dt`: Caller-supplied value used during processing.
    - `dirac`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    psi = psi + dt * dirac
    return psi

# exclude -> not need?
def _calc_gauss(x, mu, sigma):
    # Gaussian distribution for wave packet initialization
    """
    Calc gauss for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `x`, `mu`, `sigma`.
    2. Builds intermediate state such as `gauss` before applying the main logic.
    3. Delegates side effects or helper work through `exp()`.
    4. Returns the assembled result to the caller.

    Inputs:
    - `x`: Caller-supplied value used during processing.
    - `mu`: Caller-supplied value used during processing.
    - `sigma`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    gauss = exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return gauss

def calc_ckm_component(psi_, ckm_val):
    # CKM matrix rotation component for quark flavor mixing
    """
    Calc ckm component for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `psi_`, `ckm_val`.
    2. Builds intermediate state such as `ckm_component` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `psi_`: Caller-supplied value used during processing.
    - `ckm_val`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    ckm_component = psi_ * ckm_val
    return ckm_component

def _calc_quark_doublet(psi, ckm_component_sum):
    """
    Calc quark doublet for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `psi`, `ckm_component_sum`.
    2. Builds intermediate state such as `quark_doublet` before applying the main logic.
    3. Delegates side effects or helper work through `stack()`.
    4. Returns the assembled result to the caller.
    Inputs:
    - `psi`: Caller-supplied value used during processing.
    - `ckm_component_sum`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    quark_doublet = stack([psi, ckm_component_sum], axis=1)
    return quark_doublet