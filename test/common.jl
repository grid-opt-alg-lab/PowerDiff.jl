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

# =============================================================================
# Common Test Setup and Utilities
# =============================================================================
#
# Shared test infrastructure used by included test files (not runtests.jl,
# which defines its own load_test_case inline).
#
# Data loaders:
#   load_test_case  — parse a MATPOWER case into a PowerIO.Network (via PowerDiff.parse_file)
#   load_pm_case    — parse a PowerModels dictionary for oracle comparisons only
#
# Programmatic networks:
#   create_2bus_network          — minimal 2-bus DC network with known closed-form solution
#   create_3bus_congested_network — 3-bus network where line 1→3 saturates, forcing
#                                   dispatch of an expensive generator at bus 2
#
# Independent PF solver:
#   solve_pf_pq     — PQ-only Newton-Raphson AC power flow using ForwardDiff Jacobians,
#                     independent of PowerDiff's analytical sensitivity code
#   pf_residual_pq  — power mismatch residual in rectangular coordinates
#
# Convention: when test data is unavailable, tests use `@test_skip false` to register
# a visible "Broken" marker in test output rather than silently passing.
# =============================================================================

using Test
using LinearAlgebra
using SparseArrays
using Statistics
using PowerDiff
using PowerModels
using ForwardDiff
using Ipopt
using JuMP: MOI

# PowerModels test data directory and PowerDiff-owned PGLib artifact handle
const PM_DATA_DIR = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
const PD_PGLIB_DIR = PowerDiff.get_path(:pglib)

# Build PowerDiff network tables (the NamedTuple that DCNetwork/ACNetwork consume,
# see PowerDiff._network_data) directly, for programmatic test networks. Values are
# taken as-is — already normalized — like the removed hand-built ParsedCase path.
pd_bus(bus_i, bus_type; pd=0.0, qd=0.0, gs=0.0, bs=0.0, vm=1.0, va=0.0, vmin=0.9, vmax=1.1) =
    (; bus_i, bus_type, pd, qd, gs, bs, vm, va, vmin, vmax)
pd_gen(index, gen_bus; pg=0.0, qg=0.0, qmin=0.0, qmax=0.0, vg=1.0, pmin=0.0, pmax=0.0, cost=(0.0, 0.0, 0.0)) =
    (; index, gen_bus, pg, qg, qmin, qmax, vg, pmin, pmax, cost)
pd_branch(index, f_bus, t_bus; br_r, br_x, br_b=0.0, rate_a=Inf, rate_b=0.0, rate_c=0.0,
          tap=1.0, shift=0.0, angmin=-pi / 3, angmax=pi / 3) =
    (; index, f_bus, t_bus, br_r, br_x, br_b, rate_a, rate_b, rate_c, tap, shift, angmin, angmax)
pd_case(bus, gen, branch; name="case", baseMVA=100.0) = (; name, baseMVA, bus, gen, branch)

"""
    load_test_case(case_name::String)

Load a PowerModels test fixture through `PowerDiff.parse_file`, returning a `PowerIO.Network`.
"""
function load_test_case(case_name::String)
    case_path = joinpath(PM_DATA_DIR, case_name)
    if isfile(case_path)
        return PowerDiff.parse_file(case_path)
    else
        @warn "Test case not found: $case_path"
        return nothing
    end
end

"""
    load_pm_case(case_name::String; basic=false)

Load a PowerModels dictionary for oracle comparisons.
"""
function load_pm_case(case_name::String; basic::Bool=false)
    case_path = joinpath(PM_DATA_DIR, case_name)
    if isfile(case_path)
        data = PowerModels.parse_file(case_path)
        return basic ? PowerModels.make_basic_network(data) : data
    else
        return nothing
    end
end

load_raw_case(case_name::String) = load_pm_case(case_name)

"""
    load_ac_pf_state(case_name::String)

Solve an AC power flow with PowerModels, then wrap the resulting voltage vector
in PowerDiff's typed AC network representation.
"""
function load_ac_pf_state(case_name::String)
    data = load_test_case(case_name)
    pm_data = load_pm_case(case_name; basic=true)
    if isnothing(data) || isnothing(pm_data)
        return nothing
    end

    PowerModels.compute_ac_pf!(pm_data)
    v = PowerModels.calc_basic_bus_voltage(pm_data)
    return ACPowerFlowState(ACNetwork(data), v)
end

"""
    create_2bus_network(; fmax=100.0, gmax=10.0, cl=10.0, cq=0.0, tau=0.0)

Create a minimal 2-bus test network.

b=[-10.0] gives susceptance magnitude 10 p.u. (W = -b = 10), so flow = 10 * Δθ.
Generator at bus 1 (reference), load at bus 2.  With d=[0,1], the
closed-form solution is g=1, θ₂=-0.1, f=1.
"""
function create_2bus_network(; fmax=100.0, gmax=10.0, cl=10.0, cq=0.0, tau=0.0)
    n, m, k = 2, 1, 1
    A = sparse([1.0 -1.0])
    G_inc = sparse(reshape([1.0, 0.0], 2, 1))
    b = [-10.0]

    return DCNetwork(n, m, k, A, G_inc, b;
        fmax=[fmax], gmax=[gmax], gmin=[0.0],
        cl=[cl], cq=[cq], ref_bus=1, tau=tau)
end

"""
    create_3bus_congested_network()

Create a 3-bus network with congestion on line 1→3.

Topology: gen at bus 1 (cheap, cl=10) and bus 2 (expensive, cl=50),
load at bus 3. Line 1→3 saturates at fmax=0.5, forcing the expensive
generator at bus 2 to supply the remainder via line 2→3.  This creates
a non-trivial LMP split: bus 3 price exceeds bus 1 price by the
congestion rent.
"""
function create_3bus_congested_network()
    n, m, k = 3, 2, 2
    A = sparse([
        1.0  0.0 -1.0;   # Line 1: 1→3 (congested)
        0.0  1.0 -1.0    # Line 2: 2→3
    ])
    G_inc = sparse([
        1.0 0.0;   # Gen 1 at bus 1 (cheap)
        0.0 1.0;   # Gen 2 at bus 2 (expensive)
        0.0 0.0    # No gen at bus 3 (load)
    ])
    b = [-10.0, -10.0]

    return DCNetwork(n, m, k, A, G_inc, b;
        fmax=[0.5, 10.0],  # Line 1→3 constrained
        gmax=[10.0, 10.0], gmin=[0.0, 0.0],
        cl=[10.0, 50.0], cq=[0.0, 0.0],
        ref_bus=1, tau=0.0)
end

# =============================================================================
# PQ-Only Newton-Raphson Solver
#
# Solves AC power flow treating ALL non-slack buses as PQ (free voltage).
# Used by finite-difference verification tests to compare against analytical
# sensitivities, which also treat all non-slack buses as PQ.
# =============================================================================

"""
    pf_residual_pq(state, Y_re, Y_im, p_target, q_target, v_slack_re, v_slack_im, idx_slack, n)

Power flow residual in rectangular form. `state` = [v_re[non_slack]; v_im[non_slack]].
Uses standard convention: S_i = V_i * conj(I_i) where I = Y*V.
All arguments are real-valued for ForwardDiff compatibility.
"""
function pf_residual_pq(state, Y_re, Y_im, p_target, q_target,
                        v_slack_re, v_slack_im, idx_slack, n)
    d = n - 1
    non_slack = [i for i in 1:n if i != idx_slack]

    T = eltype(state)
    v_re = zeros(T, n)
    v_im = zeros(T, n)
    v_re[idx_slack] = v_slack_re
    v_im[idx_slack] = v_slack_im
    for (idx, bus) in enumerate(non_slack)
        v_re[bus] = state[idx]
        v_im[bus] = state[d + idx]
    end

    P = zeros(T, n)
    Q = zeros(T, n)
    for i in 1:n
        for k in 1:n
            I_re = Y_re[i,k]*v_re[k] - Y_im[i,k]*v_im[k]
            I_im = Y_re[i,k]*v_im[k] + Y_im[i,k]*v_re[k]
            P[i] += v_re[i]*I_re + v_im[i]*I_im
            Q[i] += v_im[i]*I_re - v_re[i]*I_im
        end
    end

    return [P[non_slack] - p_target; Q[non_slack] - q_target]
end

"""
    solve_pf_pq(Y, v_base, p_target, q_target, idx_slack; max_iter=30, tol=1e-12)

Solve AC power flow treating ALL non-slack buses as PQ (free voltage
magnitude and angle). Uses Newton-Raphson with ForwardDiff Jacobian so
the solver is fully independent of PowerDiff's analytical sensitivity
code. tol=1e-12 gives near-machine-precision convergence, ensuring FD
perturbations dominate solver noise.

Returns converged complex voltage vector.
"""
function solve_pf_pq(Y, v_base, p_target, q_target, idx_slack;
                     max_iter=30, tol=1e-12)
    n = length(v_base)
    non_slack = [i for i in 1:n if i != idx_slack]
    d = n - 1

    Y_re = real.(Matrix(Y))
    Y_im = imag.(Matrix(Y))
    v_slack_re = real(v_base[idx_slack])
    v_slack_im = imag(v_base[idx_slack])

    state = [real.(v_base[non_slack]); imag.(v_base[non_slack])]

    for _ in 1:max_iter
        r = pf_residual_pq(state, Y_re, Y_im, p_target, q_target,
                           v_slack_re, v_slack_im, idx_slack, n)
        norm(r) < tol && break
        J = ForwardDiff.jacobian(
            s -> pf_residual_pq(s, Y_re, Y_im, p_target, q_target,
                                v_slack_re, v_slack_im, idx_slack, n),
            state)
        state = state - J \ r
    end

    v = copy(v_base)
    for (idx, bus) in enumerate(non_slack)
        v[bus] = state[idx] + im * state[d + idx]
    end
    return v
end
