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

# FD verification of AC PF topology sensitivities (∂vm/∂g, ∂vm/∂b, ∂va/∂g,
# ∂va/∂b, ∂f/∂g, ∂f/∂b, ∂im/∂g, ∂im/∂b). Perturbs branch conductance/susceptance,
# re-solves Newton-Raphson, compares against analytical chain-rule formulas.
#
# This matches the ACPowerFlowState sensitivity formulation, which treats all
# non-slack buses as PQ.

using PowerDiff
using PowerModels
using ForwardDiff
using LinearAlgebra
using Test

import PowerDiff: admittance_matrix

# PQ Newton solver: pf_residual_pq / solve_pf_pq from common.jl
@isdefined(pf_residual_pq) || include("common.jl")

_branch_flows(v, net) = real.(branch_power(net, v))
_branch_currents_mag(v, net) = abs.(branch_current(net, v))

function _perturbed_voltage(state::ACPowerFlowState, param::Symbol, branch_idx::Int, epsilon::Float64,
                            p_target, q_target)
    net_pert = deepcopy(state.net)
    if param === :g
        net_pert.g[branch_idx] += epsilon
    else
        net_pert.b[branch_idx] += epsilon
    end
    Y_pert = admittance_matrix(net_pert)
    v_pert = solve_pf_pq(Y_pert, state.v, p_target, q_target, state.idx_slack)
    return v_pert, Y_pert, net_pert
end

@testset "AC PF Topology Sensitivities (:g, :b)" begin
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")

    @testset "Finite-difference verification — case5" begin
        file = joinpath(pm_path, "case5.m")
        state = load_ac_pf_state("case5.m")

        n = state.n
        m = state.m
        Y = admittance_matrix(state.net)
        non_slack = [i for i in 1:n if i != state.idx_slack]

        I_base = Y * state.v
        S_base = state.v .* conj.(I_base)
        p_base = real.(S_base)[non_slack]
        q_base = imag.(S_base)[non_slack]

        dvm_dg = calc_sensitivity(state, :vm, :g)
        dva_dg = calc_sensitivity(state, :va, :g)
        dvm_db = calc_sensitivity(state, :vm, :b)
        dva_db = calc_sensitivity(state, :va, :b)
        df_dg = calc_sensitivity(state, :f, :g)
        df_db = calc_sensitivity(state, :f, :b)
        dim_dg = calc_sensitivity(state, :im, :g)
        dim_db = calc_sensitivity(state, :im, :b)

        ε = 1e-5
        # Spot-check first and third branches for FD agreement; full size/finiteness
        # coverage is in the "Smoke tests — all 10 combinations" testset below.
        test_branches = [1, min(3, m)]

        for param in (:g, :b)
            S_vm = param === :g ? dvm_dg : dvm_db
            S_va = param === :g ? dva_dg : dva_db
            S_f = param === :g ? df_dg : df_db
            S_im = param === :g ? dim_dg : dim_db

            for e in test_branches
                @testset "∂/∂$(param)_$e" begin
                    v_p, Y_p, net_p = _perturbed_voltage(state, param, e, ε, p_base, q_base)
                    v_m, Y_m, net_m = _perturbed_voltage(state, param, e, -ε, p_base, q_base)

                    fd_vm = (abs.(v_p) - abs.(v_m)) / (2ε)
                    if norm(fd_vm) > 1e-6
                        rel_err_vm = norm(Matrix(S_vm)[:, e] - fd_vm) / norm(fd_vm)
                        @test rel_err_vm < 1e-2
                    end

                    fd_va = (angle.(v_p) - angle.(v_m)) / (2ε)
                    if norm(fd_va) > 1e-6
                        rel_err_va = norm(Matrix(S_va)[:, e] - fd_va) / norm(fd_va)
                        @test rel_err_va < 1e-2
                    end

                    fd_f = (_branch_flows(v_p, net_p) -
                            _branch_flows(v_m, net_m)) / (2ε)
                    if norm(fd_f) > 1e-6
                        rel_err_f = norm(Matrix(S_f)[:, e] - fd_f) / norm(fd_f)
                        @test rel_err_f < 1e-2
                    end

                    fd_im = (_branch_currents_mag(v_p, net_p) -
                             _branch_currents_mag(v_m, net_m)) / (2ε)
                    if norm(fd_im) > 1e-6
                        rel_err_im = norm(Matrix(S_im)[:, e] - fd_im) / norm(fd_im)
                        @test rel_err_im < 1e-2
                    end
                end
            end
        end
    end

    @testset "Transformer, phase shift, and parallel-line finite differences" begin
        buses = [
            pd_bus(1, 3; vmax=1.1, vmin=0.9),
            pd_bus(2, 1; vmax=1.1, vmin=0.9),
            pd_bus(3, 1; vmax=1.1, vmin=0.9),
        ]
        gens = [
            pd_gen(1, 1; pg=0.5, qmax=1.0, qmin=-1.0, vg=1.0, pmax=2.0, pmin=0.0, cost=(1.0, 1.0, 0.0)),
        ]
        branches = [
            pd_branch(1, 1, 2; br_r=0.01, br_x=0.10, br_b=0.02, rate_a=2.0, rate_b=2.0, rate_c=2.0, tap=1.05, shift=0.12, angmin=-π / 3, angmax=π / 3),
            pd_branch(2, 1, 2; br_r=0.02, br_x=0.20, br_b=0.01, rate_a=2.0, rate_b=2.0, rate_c=2.0, tap=1.00, shift=0.00, angmin=-π / 3, angmax=π / 3),
            pd_branch(3, 2, 3; br_r=0.01, br_x=0.15, br_b=0.03, rate_a=2.0, rate_b=2.0, rate_c=2.0, tap=0.97, shift=-0.08, angmin=-π / 3, angmax=π / 3),
        ]
        net = ACNetwork(pd_case(buses, gens, branches; name="topology_fd"))
        state = ACPowerFlowState(net, [1.01 + 0.02im, 0.98 - 0.04im, 1.02 + 0.01im])
        non_slack = [i for i in 1:state.n if i != state.idx_slack]
        injections = state.v .* conj.(state.Y * state.v)
        p_target = real.(injections)[non_slack]
        q_target = imag.(injections)[non_slack]
        ε = 1e-6

        for param in (:g, :b)
            S_vm = calc_sensitivity(state, :vm, param)
            S_va = calc_sensitivity(state, :va, param)
            S_f = calc_sensitivity(state, :f, param)
            S_im = calc_sensitivity(state, :im, param)
            for e in 1:net.m
                v_p, _, net_p = _perturbed_voltage(state, param, e, ε, p_target, q_target)
                v_m, _, net_m = _perturbed_voltage(state, param, e, -ε, p_target, q_target)
                @test Matrix(S_vm)[:, e] ≈ (abs.(v_p) - abs.(v_m)) / (2ε) atol=1e-6 rtol=1e-4
                @test Matrix(S_va)[:, e] ≈ (angle.(v_p) - angle.(v_m)) / (2ε) atol=1e-6 rtol=1e-4
                @test Matrix(S_f)[:, e] ≈ (_branch_flows(v_p, net_p) - _branch_flows(v_m, net_m)) / (2ε) atol=1e-6 rtol=1e-4
                @test Matrix(S_im)[:, e] ≈ (_branch_currents_mag(v_p, net_p) - _branch_currents_mag(v_m, net_m)) / (2ε) atol=1e-6 rtol=1e-4
            end
        end
    end

    @testset "Smoke tests — all 10 combinations" begin
        file = joinpath(pm_path, "case5.m")
        state = load_ac_pf_state("case5.m")

        n = state.n
        m = state.m

        combos = [
            (:vm, :g, (n, m)), (:va, :g, (n, m)), (:v, :g, (n, m)),
            (:f, :g, (m, m)), (:im, :g, (m, m)),
            (:vm, :b, (n, m)), (:va, :b, (n, m)), (:v, :b, (n, m)),
            (:f, :b, (m, m)), (:im, :b, (m, m)),
        ]

        for (op, param, expected_size) in combos
            @testset "$op w.r.t. $param" begin
                S = calc_sensitivity(state, op, param)
                @test S isa Sensitivity
                @test S.formulation == :acpf
                @test S.operand == op
                @test S.parameter == param
                @test size(S) == expected_size
                @test all(isfinite, Matrix(S))
            end
        end
    end

    @testset "Error: state without ACNetwork" begin
        file = joinpath(pm_path, "case5.m")
        full_state = load_ac_pf_state("case5.m")

        raw_state = ACPowerFlowState(full_state.v, full_state.Y;
            idx_slack=full_state.idx_slack, branch_data=full_state.branch_data)
        @test isnothing(raw_state.net)
        @test_throws ArgumentError calc_sensitivity(raw_state, :vm, :g)
        @test_throws ArgumentError calc_sensitivity(raw_state, :va, :b)
    end

    @testset "Non-basic network — col_to_id maps to branch IDs" begin
        file = joinpath(pm_path, "case5.m")
        state = load_ac_pf_state("case5.m")

        S = calc_sensitivity(state, :vm, :g)
        @test S.formulation == :acpf
        @test length(S.col_to_id) == state.m
        @test S.col_to_id == state.net.id_map.branch_ids
        @test all(isfinite, Matrix(S))
    end

    @testset "Sensitivity metadata" begin
        file = joinpath(pm_path, "case14.m")
        state = load_ac_pf_state("case14.m")

        for param in (:g, :b)
            S = calc_sensitivity(state, :vm, param)
            @test S.formulation == :acpf
            @test S.operand == :vm
            @test S.parameter == param
            @test size(S, 1) == state.n
            @test size(S, 2) == state.m
        end
    end
end
