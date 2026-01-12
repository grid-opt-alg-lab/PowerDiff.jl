# =============================================================================
# Sensitivity Result Types
# =============================================================================
#
# Structs for holding sensitivity computation results from DC OPF, DC power flow,
# and AC power flow analyses.

# =============================================================================
# DC Sensitivity Types
# =============================================================================

"""
    DemandSensitivity <: AbstractSensitivity

Sensitivity of DC OPF/power flow solution with respect to nodal demand.

# Fields
- `dθ_dd`: Jacobian dtheta/dd (n x n)
- `dg_dd`: Jacobian dg/dd (k x n)
- `df_dd`: Jacobian df/dd (m x n)
- `dlmp_dd`: Jacobian dLMP/dd (n x n)
"""
struct DemandSensitivity <: AbstractSensitivity
    dθ_dd::Matrix{Float64}
    dg_dd::Matrix{Float64}
    df_dd::Matrix{Float64}
    dlmp_dd::Matrix{Float64}
end

"""
    CostSensitivity <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to cost coefficients.

# Fields
- `dg_dcq`: Jacobian dg/dcq (k x k)
- `dg_dcl`: Jacobian dg/dcl (k x k)
- `dlmp_dcq`: Jacobian dLMP/dcq (n x k)
- `dlmp_dcl`: Jacobian dLMP/dcl (n x k)
"""
struct CostSensitivity <: AbstractSensitivity
    dg_dcq::Matrix{Float64}
    dg_dcl::Matrix{Float64}
    dlmp_dcq::Matrix{Float64}
    dlmp_dcl::Matrix{Float64}
end

"""
    FlowLimitSensitivity <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to line flow limits.

# Fields
- `dθ_dfmax`: Jacobian dtheta/dfmax (n x m)
- `dg_dfmax`: Jacobian dg/dfmax (k x m)
- `df_dfmax`: Jacobian df/dfmax (m x m)
- `dlmp_dfmax`: Jacobian dLMP/dfmax (n x m)
"""
struct FlowLimitSensitivity <: AbstractSensitivity
    dθ_dfmax::Matrix{Float64}
    dg_dfmax::Matrix{Float64}
    df_dfmax::Matrix{Float64}
    dlmp_dfmax::Matrix{Float64}
end

"""
    SusceptanceSensitivity <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to branch susceptances.

# Fields
- `dθ_db`: Jacobian dtheta/db (n x m)
- `dg_db`: Jacobian dg/db (k x m)
- `df_db`: Jacobian df/db (m x m)
- `dlmp_db`: Jacobian dLMP/db (n x m)
"""
struct SusceptanceSensitivity <: AbstractSensitivity
    dθ_db::Matrix{Float64}
    dg_db::Matrix{Float64}
    df_db::Matrix{Float64}
    dlmp_db::Matrix{Float64}
end

"""
    SwitchingSensitivity <: AbstractSensitivityTopology

Sensitivity of power flow or OPF solution with respect to switching states.

Works for both DCPowerFlowState (simple Laplacian derivative) and DCOPFSolution
(via KKT implicit differentiation).

# Fields
- `dθ_dz`: Jacobian dtheta/dz (n x m)
- `dg_dz`: Jacobian dg/dz (k x m), zeros for power flow
- `df_dz`: Jacobian df/dz (m x m)
- `dlmp_dz`: Jacobian dLMP/dz (n x m), zeros for power flow
"""
struct SwitchingSensitivity <: AbstractSensitivityTopology
    dθ_dz::Matrix{Float64}
    dg_dz::Matrix{Float64}
    df_dz::Matrix{Float64}
    dlmp_dz::Matrix{Float64}
end

"""
    SwitchingSensitivity(dθ_dz, df_dz)

Convenience constructor for power flow (no generation or LMP sensitivity).
"""
function SwitchingSensitivity(dθ_dz::Matrix{Float64}, df_dz::Matrix{Float64})
    n, m = size(dθ_dz)
    dg_dz = zeros(0, m)  # No generators in power flow
    dlmp_dz = zeros(n, m)  # No LMP in power flow
    SwitchingSensitivity(dθ_dz, dg_dz, df_dz, dlmp_dz)
end

# =============================================================================
# AC Voltage Sensitivity Types
# =============================================================================

"""
    VoltagePowerSensitivity <: AbstractSensitivityPower

Sensitivity of bus voltages with respect to power injections.

# Fields
- `∂v_∂p`: Complex phasor sensitivity dv/dp (n x n)
- `∂v_∂q`: Complex phasor sensitivity dv/dq (n x n)
- `∂vm_∂p`: Magnitude sensitivity d|v|/dp (n x n)
- `∂vm_∂q`: Magnitude sensitivity d|v|/dq (n x n)
"""
struct VoltagePowerSensitivity <: AbstractSensitivityPower
    ∂v_∂p::Matrix{ComplexF64}
    ∂v_∂q::Matrix{ComplexF64}
    ∂vm_∂p::Matrix{Float64}
    ∂vm_∂q::Matrix{Float64}
end

"""
    VoltageTopologySensitivity <: AbstractSensitivityTopology

Sensitivity of bus voltage magnitudes with respect to topology (admittance) parameters.

# Fields
- `∂vm_∂g`: Sensitivity d|v|/dG (n x num_edges)
- `∂vm_∂b`: Sensitivity d|v|/dB (n x num_edges)
"""
struct VoltageTopologySensitivity <: AbstractSensitivityTopology
    ∂vm_∂g::Matrix{Float64}
    ∂vm_∂b::Matrix{Float64}
end

# =============================================================================
# AC Current Sensitivity Types
# =============================================================================

"""
    CurrentPowerSensitivity <: AbstractSensitivityPower

Sensitivity of branch currents with respect to power injections.

# Fields
- `∂I_∂p`: Complex current phasor sensitivity dI/dp (m x n)
- `∂I_∂q`: Complex current phasor sensitivity dI/dq (m x n)
- `∂Im_∂p`: Current magnitude sensitivity d|I|/dp (m x n)
- `∂Im_∂q`: Current magnitude sensitivity d|I|/dq (m x n)
"""
struct CurrentPowerSensitivity <: AbstractSensitivityPower
    ∂I_∂p::Matrix{ComplexF64}
    ∂I_∂q::Matrix{ComplexF64}
    ∂Im_∂p::Matrix{Float64}
    ∂Im_∂q::Matrix{Float64}
end

"""
    CurrentTopologySensitivity <: AbstractSensitivityTopology

Sensitivity of branch currents with respect to topology (admittance) parameters.

# Fields
- `∂Im_∂g`: Sensitivity d|I|/dG (m x num_edges)
- `∂Im_∂b`: Sensitivity d|I|/dB (m x num_edges)
"""
struct CurrentTopologySensitivity <: AbstractSensitivityTopology
    ∂Im_∂g::Matrix{Float64}
    ∂Im_∂b::Matrix{Float64}
end
