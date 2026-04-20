# -----------------------------------------------------------------------------
# HIGGS FIELD DYNAMICS - ATOMIC EQUATIONS
# -----------------------------------------------------------------------------

def calc_lambda_H(mass, vev):
    # Self-coupling constant: λ_H = m^2 / (2 v^2)
    """
    Calc lambda H for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `mass`, `vev`.
    2. Builds intermediate state such as `lambda_H` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `mass`: Caller-supplied value used during processing.
    - `vev`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    lambda_H = (mass ** 2) / (2.0 * vev ** 2)
    return lambda_H

def calc_mu_sq(vev, lambda_H):
    # Squared mass parameter of the potential: μ^2 = v^2 * λ_H
    """
    Calc mu sq for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `vev`, `lambda_H`.
    2. Builds intermediate state such as `mu_sq` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `vev`: Caller-supplied value used during processing.
    - `lambda_H`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    mu_sq = (vev ** 2) * lambda_H
    return mu_sq

def calc_dV_dh(vev, lambda_H, h, mu_sq):
    # Higgs potential derivative: dV/dh = -μ^2(v+h) + λ(v+h)^3
    """
    Calc dV dh for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `vev`, `lambda_H`, `h`, `mu_sq`.
    2. Builds intermediate state such as `dV_dh` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `vev`: Caller-supplied value used during processing.
    - `lambda_H`: Caller-supplied value used during processing.
    - `h`: Caller-supplied value used during processing.
    - `mu_sq`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    dV_dh = -mu_sq * (vev + h) + lambda_H * (vev + h) ** 3
    return dV_dh

def calc_spatial_diff_h(h_, h__, d):
    # Central difference for spatial derivative: (ψ_p - ψ_m) / 2d
    """
    Calc spatial diff h for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `h_`, `h__`, `d`.
    2. Builds intermediate state such as `spatial_diff` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `h_`: Caller-supplied value used during processing.
    - `h__`: Caller-supplied value used during processing.
    - `d`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    spatial_diff = (h_ - h__) / (2.0 * d)
    return spatial_diff

def calc_time_diff_h(field_current, field_prev, dt):
    # First order backward difference for time: (ψ_t - ψ_t-dt) / dt
    """
    Calc time diff h for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `field_current`, `field_prev`, `dt`.
    2. Builds intermediate state such as `time_diff` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `field_current`: Caller-supplied value used during processing.
    - `field_prev`: Caller-supplied value used during processing.
    - `dt`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    time_diff = (field_current - field_prev) / dt
    return time_diff

def calc_laplacian_h(spatial_diff_sum):
    # Summation of second derivatives or spatial differences for scalar laplacian
    """
    Calc laplacian h for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `spatial_diff_sum`.
    2. Builds intermediate state such as `laplacian_h` before applying the main logic.
    3. Delegates side effects or helper work through `sum()`.
    4. Returns the assembled result to the caller.

    Inputs:
    - `spatial_diff_sum`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    laplacian_h = sum(spatial_diff_sum)
    return laplacian_h

def calc_mass_term_h(mass, h):
    # Mass contribution to the Klein-Gordon equation: -m^2 * h
    """
    Calc mass term h for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `mass`, `h`.
    2. Builds intermediate state such as `mass_term` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `mass`: Caller-supplied value used during processing.
    - `h`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    mass_term = -(mass ** 2) * h
    return mass_term

def calc_h(h, prev_h, dt, laplacian_h, mass_term, dV_dh):
    # Discrete second-order time evolution: h_new = 2h - h_prev + dt^2 * (∇^2 h - m^2h - dV/dh)
    """
    Calc h for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `h`, `prev_h`, `dt`, `laplacian_h`.
    2. Builds intermediate state such as `h` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `h`: Caller-supplied value used during processing.
    - `prev_h`: Caller-supplied value used during processing.
    - `dt`: Caller-supplied value used during processing.
    - `laplacian_h`: Caller-supplied value used during processing.
    - `mass_term`: Caller-supplied value used during processing.
    - `dV_dh`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    h = 2.0 * h - prev_h + (dt ** 2) * (laplacian_h + mass_term - dV_dh)
    return h

def calc_phi_component(vev, h):
    # Physical Higgs component of the doublet: (v + h) / sqrt(2)
    """
    Calc phi component for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `vev`, `h`.
    2. Builds intermediate state such as `phi_component` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `vev`: Caller-supplied value used during processing.
    - `h`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    phi_component = (vev + h) / (2.0 ** 0.5)
    return phi_component

def calc_kinetic_energy(time_diff):
    # Local kinetic energy density: 0.5 * (∂t h)^2
    """
    Calc kinetic energy for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `time_diff`.
    2. Builds intermediate state such as `kinetic_energy` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `time_diff`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    kinetic_energy = 0.5 * (time_diff ** 2)
    return kinetic_energy

def calc_gradient_energy(spatial_diff_vector):
    # Local gradient energy density: 0.5 * |∇h|^2
    """
    Calc gradient energy for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `spatial_diff_vector`.
    2. Builds intermediate state such as `gradient_energy` before applying the main logic.
    3. Delegates side effects or helper work through `sum()`.
    4. Returns the assembled result to the caller.

    Inputs:
    - `spatial_diff_vector`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    gradient_energy = 0.5 * sum(spatial_diff_vector ** 2)
    return gradient_energy

def calc_potential_energy(mass, h, vev, lambda_H):
    # Higgs potential energy density: 0.5*m^2*h^2 + λ*v*h^3 + 0.25*λ*h^4
    """
    Calc potential energy for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `mass`, `h`, `vev`, `lambda_H`.
    2. Builds intermediate state such as `potential_energy` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `mass`: Caller-supplied value used during processing.
    - `h`: Caller-supplied value used during processing.
    - `vev`: Caller-supplied value used during processing.
    - `lambda_H`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    potential_energy = 0.5 * (mass ** 2) * (h ** 2) + lambda_H * vev * (h ** 3) + 0.25 * lambda_H * (h ** 4)
    return potential_energy

def calc_energy_density(kinetic_energy, gradient_energy, potential_energy):
    # Total energy density of the scalar field
    """
    Calc energy density for the surrounding workflow.

    Workflow:
    1. Reads and normalizes the incoming inputs, including `kinetic_energy`, `gradient_energy`, `potential_energy`.
    2. Builds intermediate state such as `energy_density` before applying the main logic.
    3. Returns the assembled result to the caller.

    Inputs:
    - `kinetic_energy`: Caller-supplied value used during processing.
    - `gradient_energy`: Caller-supplied value used during processing.
    - `potential_energy`: Caller-supplied value used during processing.

    Returns:
    - Returns the computed result for the caller.
    """
    energy_density = kinetic_energy + gradient_energy + potential_energy
    return energy_density