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

# Voltage-Power Sensitivity Analysis for AC Power Flow
#
# Computes analytical sensitivity coefficients ∂v/∂p and ∂v/∂q using
# implicit differentiation on the AC power flow equations.
#
# Reference:
# K. Christakou, et al., "Efficient Computation of Sensitivity Coefficients
# of Node Voltages and Line Currents in Unbalanced Radial Electrical
# Distribution Networks", IEEE Trans. Smart Grid, vol. 4, no. 2, pp. 741-750, 2013.

# =============================================================================
# Voltage-Power Sensitivity
# =============================================================================

"""
    calc_voltage_power_sensitivities(v, Y; idx_slack=1, full=true)

Compute sensitivity of bus voltages with respect to active and reactive power injections.

Uses implicit differentiation on the AC power flow equations. The power flow
equations in rectangular coordinates are linearized around the operating point
to compute the sensitivity matrices.

# Arguments
- `v::Vector{ComplexF64}`: Voltage phasors at all buses
- `Y::AbstractMatrix{ComplexF64}`: Bus admittance matrix

# Keyword Arguments
- `idx_slack::Int=1`: Index of the slack (reference) bus
- `full::Bool=true`: If true, include zero rows/columns for slack bus

# Returns
NamedTuple with fields:
- `dv_dp`: Complex phasor voltage sensitivity to active power (n × n)
- `dv_dq`: Complex phasor voltage sensitivity to reactive power (n × n)
- `dvm_dp`: Voltage magnitude sensitivity to active power (n × n)
- `dvm_dq`: Voltage magnitude sensitivity to reactive power (n × n)
- `dva_dp`: Voltage angle sensitivity to active power (n × n)
- `dva_dq`: Voltage angle sensitivity to reactive power (n × n)

# Example
```julia
state = ACPowerFlowState(pm_net)
sens = calc_sensitivity(state, :vm, :p)
# How does voltage at bus 3 change when active power at bus 2 increases?
dvdp = sens[3, 2]
```
"""
function calc_voltage_power_sensitivities(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64};
    idx_slack::Int=1,
    full::Bool=true
)
    A = _build_voltage_sensitivity_matrix(v, Y, idx_slack)
    ns = _non_slack_indices(length(v), idx_slack)
    v_ = v[ns]
    d = length(v_)
    A_lu = lu(A)
    ∂v_∂p, ∂vm_∂p, ∂va_∂p = _solve_voltage_sensitivities(A_lu, v_, d, 0)
    ∂v_∂q, ∂vm_∂q, ∂va_∂q = _solve_voltage_sensitivities(A_lu, v_, d, d)
    if full
        ∂v_∂p = _insert_slack_zeros(∂v_∂p, idx_slack, ComplexF64)
        ∂vm_∂p = _insert_slack_zeros(∂vm_∂p, idx_slack, Float64)
        ∂va_∂p = _insert_slack_zeros(∂va_∂p, idx_slack, Float64)
        ∂v_∂q = _insert_slack_zeros(∂v_∂q, idx_slack, ComplexF64)
        ∂vm_∂q = _insert_slack_zeros(∂vm_∂q, idx_slack, Float64)
        ∂va_∂q = _insert_slack_zeros(∂va_∂q, idx_slack, Float64)
    end
    return (dv_dp=∂v_∂p, dv_dq=∂v_∂q, dvm_dp=∂vm_∂p, dvm_dq=∂vm_∂q,
            dva_dp=∂va_∂p, dva_dq=∂va_∂q)
end

"""
    calc_voltage_power_sensitivities(net::Dict; full=true)

Compute voltage-power sensitivities from a solved PowerModels network.

Accepts both basic and non-basic networks. For non-basic networks, constructs
an ACPowerFlowState internally which handles ID translation.
"""
function calc_voltage_power_sensitivities(net::Dict; full::Bool=true)
    state = ACPowerFlowState(net)
    return calc_voltage_power_sensitivities(state; full=full)
end

"""
    calc_voltage_power_sensitivities(state::ACPowerFlowState; full=true)

Compute voltage-power sensitivities from an ACPowerFlowState.

This method provides a unified interface consistent with DC OPF sensitivities.
"""
function calc_voltage_power_sensitivities(state::ACPowerFlowState; full::Bool=true)
    return calc_voltage_power_sensitivities(state.v, state.Y; idx_slack=state.idx_slack, full=full)
end

"""
    calc_voltage_active_power_sensitivities(v, Y; idx_slack=1, full=true)

Compute sensitivity of bus voltages with respect to active power injections.

# Returns
Tuple (∂v_∂p, ∂vm_∂p, ∂va_∂p) where:
- `∂v_∂p`: Complex phasor sensitivity ∂v/∂p (n × n or (n-1) × (n-1) if full=false)
- `∂vm_∂p`: Magnitude sensitivity ∂|v|/∂p
- `∂va_∂p`: Angle sensitivity ∂θ/∂p
"""
function calc_voltage_active_power_sensitivities(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64};
    idx_slack::Int=1,
    full::Bool=true
)
    A = _build_voltage_sensitivity_matrix(v, Y, idx_slack)
    ns = _non_slack_indices(length(v), idx_slack)
    v_ = v[ns]
    d = length(v_)
    A_lu = lu(A)
    ∂v_∂p, ∂vm_∂p, ∂va_∂p = _solve_voltage_sensitivities(A_lu, v_, d, 0)
    if full
        ∂v_∂p = _insert_slack_zeros(∂v_∂p, idx_slack, ComplexF64)
        ∂vm_∂p = _insert_slack_zeros(∂vm_∂p, idx_slack, Float64)
        ∂va_∂p = _insert_slack_zeros(∂va_∂p, idx_slack, Float64)
    end
    return ∂v_∂p, ∂vm_∂p, ∂va_∂p
end

"""
    calc_voltage_reactive_power_sensitivities(v, Y; idx_slack=1, full=true)

Compute sensitivity of bus voltages with respect to reactive power injections.

# Returns
Tuple (∂v_∂q, ∂vm_∂q, ∂va_∂q) where:
- `∂v_∂q`: Complex phasor sensitivity ∂v/∂q
- `∂vm_∂q`: Magnitude sensitivity ∂|v|/∂q
- `∂va_∂q`: Angle sensitivity ∂θ/∂q
"""
function calc_voltage_reactive_power_sensitivities(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64};
    idx_slack::Int=1,
    full::Bool=true
)
    A = _build_voltage_sensitivity_matrix(v, Y, idx_slack)
    ns = _non_slack_indices(length(v), idx_slack)
    v_ = v[ns]
    d = length(v_)
    A_lu = lu(A)
    ∂v_∂q, ∂vm_∂q, ∂va_∂q = _solve_voltage_sensitivities(A_lu, v_, d, d)
    if full
        ∂v_∂q = _insert_slack_zeros(∂v_∂q, idx_slack, ComplexF64)
        ∂vm_∂q = _insert_slack_zeros(∂vm_∂q, idx_slack, Float64)
        ∂va_∂q = _insert_slack_zeros(∂va_∂q, idx_slack, Float64)
    end
    return ∂v_∂q, ∂vm_∂q, ∂va_∂q
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _solve_voltage_sensitivities(A_lu, v_, d, rhs_offset)

Shared inner loop for voltage sensitivity computation.

Solves the pre-factorized system `A_lu \\ b` for each bus k, where b has a unit
perturbation at position `rhs_offset + k`. For active power sensitivities,
`rhs_offset = 0`; for reactive power, `rhs_offset = d`.

Accepts a pre-computed LU factorization to avoid re-factorizing for each RHS.
Returns (∂v, ∂vm, ∂va) matrices of size (d × d).
"""
function _solve_voltage_sensitivities(A_lu, v_, d, rhs_offset)
    ∂v = Matrix{ComplexF64}(undef, d, d)
    ∂vm = Matrix{Float64}(undef, d, d)
    ∂va = Matrix{Float64}(undef, d, d)
    b = zeros(2d)
    x = Vector{Float64}(undef, 2d)
    # Hoist v_safe and derived vectors outside the loop — they depend only on v_, not k
    v_safe = ifelse.(abs.(v_) .> eps(Float64), v_, one(ComplexF64))
    abs_v_safe = abs.(v_safe)
    abs2_v_safe = abs2.(v_safe)
    conj_v_safe = conj.(v_safe)
    for k in 1:d
        if abs(v_[k]) > 1e-6
            fill!(b, 0.0)
            b[rhs_offset + k] = 1.0
            ldiv!(x, A_lu, b)
            if any(!isfinite, x)
                error("Voltage sensitivity solve produced non-finite values at bus $k. " *
                      "The Jacobian may be near-singular.")
            end
            ∂v[:, k] = x[1:d] + im * x[d+1:2d]
            ∂vm[:, k] = real.(∂v[:, k] .* conj_v_safe) ./ abs_v_safe
            ∂va[:, k] = imag.(∂v[:, k] .* conj_v_safe) ./ abs2_v_safe
        else
            ∂v[:, k] .= 0; ∂vm[:, k] .= 0; ∂va[:, k] .= 0
        end
    end
    return ∂v, ∂vm, ∂va
end

"""
    _build_voltage_sensitivity_matrix(v, Y, idx_slack)

Build the linearized system matrix A for voltage-power sensitivity computation.

The matrix A relates perturbations in (v_re, v_im) to perturbations in (p, q)
based on the standard-convention power flow equations:

    p + jq = V · conj(Y · V) = conj(conj(V) · Y · V)
    p = Re(conj(V) · Y · V)
    q = -Im(conj(V) · Y · V)   [standard convention]

The matrix has structure:

    A = [∂P/∂v_re   ∂P/∂v_im]
        [∂Q/∂v_re   ∂Q/∂v_im]

evaluated at the operating point (v, Y).
"""
function _build_voltage_sensitivity_matrix(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64},
    idx_slack::Int
)
    # Remove slack bus
    ns = _non_slack_indices(length(v), idx_slack)
    v_ = v[ns]
    Y_ = Y[ns, ns]

    # Current injection at non-slack buses: I = Y * v
    I = (Y * v)[ns]

    d = length(v_)

    # H = Diag(conj(v)) * Y (used in power flow Jacobian)
    H = Diagonal(conj.(v_)) * Y_

    # Build the 2d × 2d Jacobian matrix (dense: every element is filled below)
    A = zeros(2d, 2d)

    for i in 1:d
        for j in 1:d
            if i == j
                # Diagonal blocks include current injection terms
                # Top block: ∂P/∂(v_re, v_im) — unchanged
                A[i, j] = real(I[i]) + real(H[i, j])
                A[i, d+j] = imag(I[i]) - imag(H[i, j])
                # Bottom block: ∂Q/∂(v_re, v_im) — negated from Im(conj(V)·Y·V)
                A[d+i, j] = -(imag(I[i]) + imag(H[i, j]))
                A[d+i, d+j] = -(real(H[i, j]) - real(I[i]))
            else
                # Off-diagonal blocks
                A[i, j] = real(H[i, j])
                A[i, d+j] = -imag(H[i, j])
                A[d+i, j] = -imag(H[i, j])
                A[d+i, d+j] = -real(H[i, j])
            end
        end
    end

    return A
end

"""
Insert zero rows and columns for the slack bus into a reduced matrix.
"""
function _insert_slack_zeros(K::Matrix{T}, idx_slack::Int, ::Type{T}) where T
    d = size(K, 1)
    n = d + 1

    K_full = zeros(T, n, n)

    # Copy values, skipping slack index
    row_idx = 1
    for i in 1:n
        if i == idx_slack
            continue
        end
        col_idx = 1
        for j in 1:n
            if j == idx_slack
                continue
            end
            K_full[i, j] = K[row_idx, col_idx]
            col_idx += 1
        end
        row_idx += 1
    end

    return K_full
end

"""Return indices 1:n excluding idx, for removing the slack bus from vectors/matrices."""
_non_slack_indices(n::Int, idx::Int) = [1:idx-1; idx+1:n]
