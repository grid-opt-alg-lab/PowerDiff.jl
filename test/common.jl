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

using Test
using LinearAlgebra
using SparseArrays
using Statistics
using PowerDiff
using PowerModels
using ForwardDiff
using Ipopt
using JuMP: MOI

# PowerModels test data directory
const PM_DATA_DIR = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")

"""
    load_test_case(case_name::String)

Load and prepare a PowerModels test case.
Returns a basic network dictionary or nothing if not found.
"""
function load_test_case(case_name::String)
    case_path = joinpath(PM_DATA_DIR, case_name)
    if isfile(case_path)
        raw = PowerModels.parse_file(case_path)
        return PowerModels.make_basic_network(raw)
    else
        @warn "Test case not found: $case_path"
        return nothing
    end
end

"""
    load_raw_case(case_name::String)

Load a raw (non-basic) PowerModels network.
"""
function load_raw_case(case_name::String)
    case_path = joinpath(PM_DATA_DIR, case_name)
    if isfile(case_path)
        return PowerModels.parse_file(case_path)
    else
        return nothing
    end
end

"""
    create_2bus_network(; fmax=100.0, gmax=10.0, cl=10.0, cq=0.0, tau=0.0)

Create a minimal 2-bus test network.
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

Solve AC power flow treating ALL non-slack buses as PQ.
Uses Newton-Raphson with ForwardDiff Jacobian for independence from analytical code.
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
