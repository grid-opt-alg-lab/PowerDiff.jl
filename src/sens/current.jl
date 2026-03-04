# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Current-Power Sensitivity Analysis for AC Power Flow
#
# Computes analytical sensitivity coefficients ∂I/∂p and ∂I/∂q for branch currents
# using the chain rule through voltage sensitivities.
#
# Reference:
# K. Christakou, et al., "Efficient Computation of Sensitivity Coefficients
# of Node Voltages and Line Currents in Unbalanced Radial Electrical
# Distribution Networks", IEEE Trans. Smart Grid, vol. 4, no. 2, pp. 741-750, 2013.

# =============================================================================
# Current-Power Sensitivity
# =============================================================================

"""
    calc_current_power_sensitivities(v, Y, branch_data; idx_slack=1, full=true)

Compute sensitivity of branch currents with respect to active and reactive power injections.

Uses the chain rule through voltage sensitivities:
    ∂I_ℓ/∂p = Y_ij * (∂v_i/∂p - ∂v_j/∂p)

where I_ℓ is the current on branch ℓ connecting buses i and j.

# Arguments
- `v::Vector{ComplexF64}`: Voltage phasors at all buses
- `Y::AbstractMatrix{ComplexF64}`: Bus admittance matrix
- `branch_data::Dict`: PowerModels branch dictionary

# Keyword Arguments
- `idx_slack::Int=1`: Index of the slack (reference) bus
- `full::Bool=true`: If true, include zero columns for slack bus

# Returns
NamedTuple with fields:
- `dI_dp`: Complex current phasor sensitivity to active power (m × n)
- `dI_dq`: Complex current phasor sensitivity to reactive power (m × n)
- `dIm_dp`: Current magnitude sensitivity to active power (m × n)
- `dIm_dq`: Current magnitude sensitivity to reactive power (m × n)

# Example
```julia
state = ACPowerFlowState(pm_net)
sens = calc_sensitivity(state, :im, :p)
# How does current on line 2 change when active power at bus 3 increases?
dI_dp = sens[2, 3]
```
"""
function calc_current_power_sensitivities(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64},
    branch_data::Dict{String,<:Any};
    idx_slack::Int=1,
    full::Bool=true
)
    n = length(v)
    m = length(branch_data)

    # First, compute voltage-power sensitivities (always with full=true for indexing)
    ∂v_∂p, _, _ = calc_voltage_active_power_sensitivities(v, Y; idx_slack=idx_slack, full=true)
    ∂v_∂q, _, _ = calc_voltage_reactive_power_sensitivities(v, Y; idx_slack=idx_slack, full=true)

    # Initialize current sensitivity matrices
    ∂I_∂p = zeros(ComplexF64, m, n)
    ∂I_∂q = zeros(ComplexF64, m, n)
    ∂Im_∂p = zeros(Float64, m, n)
    ∂Im_∂q = zeros(Float64, m, n)

    # Compute current sensitivities for each branch using chain rule
    for (_, br) in branch_data
        ℓ = br["index"]
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]

        # Series admittance of the branch (from admittance matrix)
        Y_ft = Y[f_bus, t_bus]

        # Branch current: I_ℓ = Y_ft * (v_f - v_t)
        I_ℓ = Y_ft * (v[f_bus] - v[t_bus])

        for i in 1:n
            if i != idx_slack
                # Chain rule: ∂I_ℓ/∂p_i = Y_ft * (∂v_f/∂p_i - ∂v_t/∂p_i)
                ∂I_∂p[ℓ, i] = Y_ft * (∂v_∂p[f_bus, i] - ∂v_∂p[t_bus, i])
                ∂I_∂q[ℓ, i] = Y_ft * (∂v_∂q[f_bus, i] - ∂v_∂q[t_bus, i])

                # Magnitude sensitivity: ∂|I|/∂p = Re(∂I/∂p * conj(I)) / |I|
                if abs(I_ℓ) > 1e-6
                    ∂Im_∂p[ℓ, i] = real(∂I_∂p[ℓ, i] * conj(I_ℓ)) / abs(I_ℓ)
                    ∂Im_∂q[ℓ, i] = real(∂I_∂q[ℓ, i] * conj(I_ℓ)) / abs(I_ℓ)
                end
            end
        end
    end

    return (dI_dp=∂I_∂p, dI_dq=∂I_∂q, dIm_dp=∂Im_∂p, dIm_dq=∂Im_∂q)
end

"""
    calc_current_power_sensitivities(net::Dict; full=true)

Compute current-power sensitivities from a solved PowerModels network.

Accepts both basic and non-basic networks. For non-basic networks, constructs
an ACPowerFlowState internally which handles ID translation.
"""
function calc_current_power_sensitivities(net::Dict; full::Bool=true)
    state = ACPowerFlowState(net)
    @assert !isnothing(state.branch_data) "Failed to extract branch data from network"
    return calc_current_power_sensitivities(state; full=full)
end

"""
    calc_current_power_sensitivities(state::ACPowerFlowState; full=true)

Compute current-power sensitivities from an ACPowerFlowState.

This method provides a unified interface consistent with DC OPF sensitivities.
Requires that `state.branch_data` is not nothing.
"""
function calc_current_power_sensitivities(state::ACPowerFlowState; full::Bool=true)
    @assert !isnothing(state.branch_data) "ACPowerFlowState must have branch_data for current sensitivities"
    return calc_current_power_sensitivities(state.v, state.Y, state.branch_data; idx_slack=state.idx_slack, full=full)
end

# =============================================================================
# Branch Active Power Flow Sensitivity
# =============================================================================

"""
    calc_branch_flow_power_sensitivities(state::ACPowerFlowState)

Compute sensitivity of branch active power flows w.r.t. power injections.

Uses product rule: P_ℓ = Re(v_f · conj(I_ℓ)), so
    ∂P_ℓ/∂p_k = Re(∂v_f/∂p_k · conj(I_ℓ) + v_f · conj(∂I_ℓ/∂p_k))

Requires `state.branch_data` to be set.

# Returns
NamedTuple with:
- `df_dp`: ∂P_flow/∂p (m × n)
- `df_dq`: ∂P_flow/∂q (m × n)
"""
function calc_branch_flow_power_sensitivities(state::ACPowerFlowState)
    @assert !isnothing(state.branch_data) "ACPowerFlowState must have branch_data for flow sensitivities"

    v = state.v
    Y = state.Y
    n = state.n
    m = state.m

    # Get complex voltage sensitivities (full=true for indexing)
    ∂v_∂p, _, _ = calc_voltage_active_power_sensitivities(v, Y; idx_slack=state.idx_slack, full=true)
    ∂v_∂q, _, _ = calc_voltage_reactive_power_sensitivities(v, Y; idx_slack=state.idx_slack, full=true)

    # Get complex current sensitivities
    cur_sens = calc_current_power_sensitivities(state; full=true)
    ∂I_∂p = cur_sens.dI_dp
    ∂I_∂q = cur_sens.dI_dq

    df_dp = zeros(Float64, m, n)
    df_dq = zeros(Float64, m, n)

    for (_, br) in state.branch_data
        ℓ = br["index"]
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]

        Y_ft = Y[f_bus, t_bus]
        I_ℓ = Y_ft * (v[f_bus] - v[t_bus])
        v_f = v[f_bus]

        for k in 1:n
            # Product rule: ∂P_ℓ/∂p_k = Re(∂v_f/∂p_k · conj(I_ℓ) + v_f · conj(∂I_ℓ/∂p_k))
            df_dp[ℓ, k] = real(∂v_∂p[f_bus, k] * conj(I_ℓ) + v_f * conj(∂I_∂p[ℓ, k]))
            df_dq[ℓ, k] = real(∂v_∂q[f_bus, k] * conj(I_ℓ) + v_f * conj(∂I_∂q[ℓ, k]))
        end
    end

    return (df_dp=df_dp, df_dq=df_dq)
end

