# =============================================================================
# Sensitivity Result Types
# =============================================================================
#
# State-specific sensitivity result types for DC power flow, DC OPF, and AC power flow.
# Each state type has its own result type containing only the relevant fields.

# =============================================================================
# DC Power Flow Sensitivity Types (no generation dispatch, no LMPs)
# =============================================================================

"""
    DCPFDemandSens <: AbstractSensitivity

Sensitivity of DC power flow solution with respect to nodal demand.
Only contains angle and flow sensitivities (no generation or LMP - those don't exist for PF).

# Fields
- `dva_dd`: Jacobian d(va)/dd (n x n) - voltage angles w.r.t. demand
- `df_dd`: Jacobian df/dd (m x n) - branch flows w.r.t. demand
"""
struct DCPFDemandSens <: AbstractSensitivity
    dva_dd::Matrix{Float64}
    df_dd::Matrix{Float64}
end

"""
    DCPFSwitchingSens <: AbstractSensitivity

Sensitivity of DC power flow solution with respect to switching states.
Only contains angle and flow sensitivities (no generation or LMP - those don't exist for PF).

# Fields
- `dva_dz`: Jacobian d(va)/dz (n x m) - voltage angles w.r.t. switching
- `df_dz`: Jacobian df/dz (m x m) - branch flows w.r.t. switching
"""
struct DCPFSwitchingSens <: AbstractSensitivity
    dva_dz::Matrix{Float64}
    df_dz::Matrix{Float64}
end

# =============================================================================
# DC OPF Sensitivity Types (has generation dispatch and LMPs)
# =============================================================================

"""
    OPFDemandSens <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to nodal demand.
Contains all sensitivities including generation and LMP (from dual variables).

# Fields
- `dva_dd`: Jacobian d(va)/dd (n x n)
- `dg_dd`: Jacobian dg/dd (k x n) - generation w.r.t. demand
- `df_dd`: Jacobian df/dd (m x n)
- `dlmp_dd`: Jacobian dLMP/dd (n x n) - locational marginal prices w.r.t. demand
"""
struct OPFDemandSens <: AbstractSensitivity
    dva_dd::Matrix{Float64}
    dg_dd::Matrix{Float64}
    df_dd::Matrix{Float64}
    dlmp_dd::Matrix{Float64}
end

"""
    OPFSwitchingSens <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to switching states.

# Fields
- `dva_dz`: Jacobian d(va)/dz (n x m)
- `dg_dz`: Jacobian dg/dz (k x m)
- `df_dz`: Jacobian df/dz (m x m)
- `dlmp_dz`: Jacobian dLMP/dz (n x m)
"""
struct OPFSwitchingSens <: AbstractSensitivity
    dva_dz::Matrix{Float64}
    dg_dz::Matrix{Float64}
    df_dz::Matrix{Float64}
    dlmp_dz::Matrix{Float64}
end

"""
    OPFCostSens <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to cost coefficients.

# Fields
- `dg_dcq`: Jacobian dg/dcq (k x k) - generation w.r.t. quadratic cost
- `dg_dcl`: Jacobian dg/dcl (k x k) - generation w.r.t. linear cost
- `dlmp_dcq`: Jacobian dLMP/dcq (n x k)
- `dlmp_dcl`: Jacobian dLMP/dcl (n x k)
"""
struct OPFCostSens <: AbstractSensitivity
    dg_dcq::Matrix{Float64}
    dg_dcl::Matrix{Float64}
    dlmp_dcq::Matrix{Float64}
    dlmp_dcl::Matrix{Float64}
end

"""
    OPFFlowLimitSens <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to line flow limits.

# Fields
- `dva_dfmax`: Jacobian d(va)/dfmax (n x m)
- `dg_dfmax`: Jacobian dg/dfmax (k x m)
- `df_dfmax`: Jacobian df/dfmax (m x m)
- `dlmp_dfmax`: Jacobian dLMP/dfmax (n x m)
"""
struct OPFFlowLimitSens <: AbstractSensitivity
    dva_dfmax::Matrix{Float64}
    dg_dfmax::Matrix{Float64}
    df_dfmax::Matrix{Float64}
    dlmp_dfmax::Matrix{Float64}
end

"""
    OPFSusceptanceSens <: AbstractSensitivity

Sensitivity of DC OPF solution with respect to branch susceptances.

# Fields
- `dva_db`: Jacobian d(va)/db (n x m)
- `dg_db`: Jacobian dg/db (k x m)
- `df_db`: Jacobian df/db (m x m)
- `dlmp_db`: Jacobian dLMP/db (n x m)
"""
struct OPFSusceptanceSens <: AbstractSensitivity
    dva_db::Matrix{Float64}
    dg_db::Matrix{Float64}
    df_db::Matrix{Float64}
    dlmp_db::Matrix{Float64}
end

# =============================================================================
# Legacy DC Sensitivity Types (Deprecated)
# =============================================================================

"""
    DemandSensitivity <: AbstractSensitivity

DEPRECATED: Use `DCPFDemandSens` for power flow or `OPFDemandSens` for OPF instead.

Generic sensitivity of DC OPF/power flow solution with respect to nodal demand.
This type has fields that may not be meaningful for all state types.
"""
struct DemandSensitivity <: AbstractSensitivity
    dθ_dd::Matrix{Float64}
    dg_dd::Matrix{Float64}
    df_dd::Matrix{Float64}
    dlmp_dd::Matrix{Float64}
end

"""
    CostSensitivity <: AbstractSensitivity

DEPRECATED: Use `OPFCostSens` instead.
"""
struct CostSensitivity <: AbstractSensitivity
    dg_dcq::Matrix{Float64}
    dg_dcl::Matrix{Float64}
    dlmp_dcq::Matrix{Float64}
    dlmp_dcl::Matrix{Float64}
end

"""
    FlowLimitSensitivity <: AbstractSensitivity

DEPRECATED: Use `OPFFlowLimitSens` instead.
"""
struct FlowLimitSensitivity <: AbstractSensitivity
    dθ_dfmax::Matrix{Float64}
    dg_dfmax::Matrix{Float64}
    df_dfmax::Matrix{Float64}
    dlmp_dfmax::Matrix{Float64}
end

"""
    SusceptanceSensitivity <: AbstractSensitivity

DEPRECATED: Use `OPFSusceptanceSens` instead.
"""
struct SusceptanceSensitivity <: AbstractSensitivity
    dθ_db::Matrix{Float64}
    dg_db::Matrix{Float64}
    df_db::Matrix{Float64}
    dlmp_db::Matrix{Float64}
end

"""
    SwitchingSensitivity <: AbstractSensitivityTopology

DEPRECATED: Use `DCPFSwitchingSens` for power flow or `OPFSwitchingSens` for OPF instead.
"""
struct SwitchingSensitivity <: AbstractSensitivityTopology
    dθ_dz::Matrix{Float64}
    dg_dz::Matrix{Float64}
    df_dz::Matrix{Float64}
    dlmp_dz::Matrix{Float64}
end

"""
    SwitchingSensitivity(dθ_dz, df_dz)

DEPRECATED convenience constructor for power flow.
"""
function SwitchingSensitivity(dθ_dz::Matrix{Float64}, df_dz::Matrix{Float64})
    n, m = size(dθ_dz)
    dg_dz = zeros(0, m)
    dlmp_dz = zeros(n, m)
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
