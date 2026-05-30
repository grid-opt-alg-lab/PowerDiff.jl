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

# AC Power Flow Topology Sensitivity Analysis
#
# Computes analytical sensitivity coefficients of bus voltages and derived
# quantities (|V|, θ, |I|, P_flow) with respect to branch conductance (g)
# and susceptance (b) parameters.
#
# Uses implicit differentiation: the AC power flow equations F(v, g/b) = 0
# are linearized as J_v · dv = -∂F/∂(g or b), where J_v is the standard
# voltage-power Jacobian (same LHS as voltage.jl) and the RHS is computed
# analytically from the admittance structure Y = A' * Diag(g + jb) * A.

# =============================================================================
# RHS Computation
# =============================================================================

"""
    _build_topology_rhs(state::ACPowerFlowState, param::Symbol) → Matrix{Float64}

Build the right-hand side matrix for topology sensitivity computation.

For parameter `param ∈ {:g, :b}`, computes ∂[P;Q]/∂param as a `2d × m` matrix
where `d = n - 1` (non-slack buses) and `m` is the number of branches.

The derivation uses `Y = A' * Diag(g + jb) * A`, so `∂Y/∂g_e` is the rank-1
update `A'[:,e] * A[e,:]`. The key intermediate is:

    M[i,e] = conj(V_i) * A'[i,e] * ΔV_e

where `ΔV = A * V` are edge voltage drops. Then:
- `:g` → `∂P/∂g = Re(M)`, `∂Q/∂g = -Im(M)`
- `:b` → `∂P/∂b = -Im(M)`, `∂Q/∂b = -Re(M)`
"""
function _build_topology_rhs(state::ACPowerFlowState, param::Symbol)
    param in (:g, :b) || throw(ArgumentError("Topology parameter must be :g or :b, got :$param"))
    net = state.net
    isnothing(net) && throw(ArgumentError(
        "ACPowerFlowState must have an ACNetwork (net field) for topology " *
        "sensitivities (:g, :b). Construct via ACPowerFlowState(net::ACNetwork, v)."))

    n = state.n
    m = state.m
    v = state.v
    ns = _non_slack_indices(n, state.idx_slack)
    d = length(ns)
    RHS = Matrix{Float64}(undef, 2d, m)
    fill!(RHS, 0.0)
    reduced_idx = Dict(bus => i for (i, bus) in enumerate(ns))
    for l in 1:m
        fb, tb = net.f_bus[l], net.t_bus[l]
        tap = net.tap[l] * cis(net.shift[l])
        scale = net.sw[l] * (param === :g ? 1.0 : im)
        dYff = scale / abs2(tap)
        dYft = -scale / conj(tap)
        dYtf = -scale / tap
        dYtt = scale
        if haskey(reduced_idx, fb)
            i = reduced_idx[fb]
            value = conj(v[fb]) * (dYff * v[fb] + dYft * v[tb])
            RHS[i, l] = real(value)
            RHS[d + i, l] = -imag(value)
        end
        if haskey(reduced_idx, tb)
            i = reduced_idx[tb]
            value = conj(v[tb]) * (dYtf * v[fb] + dYtt * v[tb])
            RHS[i, l] = real(value)
            RHS[d + i, l] = -imag(value)
        end
    end
    return RHS
end

# =============================================================================
# Topology Sensitivity Solver
# =============================================================================

"""
    _solve_topology_sensitivities(state::ACPowerFlowState, param::Symbol)
        → (dv, dvm, dva)

Solve for voltage sensitivities w.r.t. branch conductance (:g) or susceptance (:b).

Returns full n × m matrices (with zero row for slack bus):
- `dv`: Complex voltage phasor sensitivity (n × m, ComplexF64)
- `dvm`: Voltage magnitude sensitivity (n × m, Float64)
- `dva`: Voltage angle sensitivity (n × m, Float64)
"""
function _solve_topology_sensitivities(state::ACPowerFlowState, param::Symbol)
    n = state.n
    m = state.m
    ns = _non_slack_indices(n, state.idx_slack)
    d = length(ns)
    v_ = state.v[ns]

    # Build and factorize the LHS (same Jacobian as voltage-power sensitivity)
    A_mat = _build_voltage_sensitivity_matrix(state.v, state.Y, state.idx_slack)
    A_lu = try
        lu(A_mat)
    catch e
        e isa LinearAlgebra.SingularException || rethrow(e)
        error("Voltage-power Jacobian is singular in topology sensitivity computation. " *
              "This typically indicates voltage collapse, a disconnected subnetwork, " *
              "or a degenerate operating point (e.g., zero-voltage buses).")
    end

    # Build RHS: 2d × m
    RHS = _build_topology_rhs(state, param)

    # Batched solve: X = -(A_lu \ RHS), size 2d × m
    X = -(A_lu \ RHS)

    # Extract complex voltage perturbation (reduced, d × m)
    dv_r = X[1:d, :] + im * X[d+1:2d, :]

    # Project to magnitude and angle, zeroing out de-energized buses
    v_safe = ifelse.(abs.(v_) .> eps(Float64), v_, one(ComplexF64))
    abs_v = abs.(v_safe)
    abs2_v = abs2.(v_safe)
    conj_v = conj.(v_safe)

    dvm_r = real.(dv_r .* conj_v) ./ abs_v
    dva_r = imag.(dv_r .* conj_v) ./ abs2_v

    # Zero out rows for de-energized buses (|V| below threshold)
    for k in 1:d
        if abs(v_[k]) <= VOLTAGE_ZERO_TOL
            dv_r[k, :] .= 0
            dvm_r[k, :] .= 0
            dva_r[k, :] .= 0
        end
    end

    # Insert zero rows for slack bus
    dv = _insert_slack_zero_rows(dv_r, state.idx_slack)
    dvm = _insert_slack_zero_rows(dvm_r, state.idx_slack)
    dva = _insert_slack_zero_rows(dva_r, state.idx_slack)

    return dv, dvm, dva
end
