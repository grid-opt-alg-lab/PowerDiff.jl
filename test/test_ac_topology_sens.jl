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

function _branch_flows(v, Y, branch_data)
    pf = zeros(length(branch_data))
    for (_, br) in branch_data
        ℓ = br["index"]
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]
        Y_ft = Y[f_bus, t_bus]
        I_ℓ = Y_ft * (v[f_bus] - v[t_bus])
        pf[ℓ] = real(v[f_bus] * conj(I_ℓ))
    end
    return pf
end

function _branch_currents_mag(v, Y, branch_data)
    im_vec = zeros(length(branch_data))
    for (_, br) in branch_data
        ℓ = br["index"]
        f_bus = br["f_bus"]
        t_bus = br["t_bus"]
        Y_ft = Y[f_bus, t_bus]
        im_vec[ℓ] = abs(Y_ft * (v[f_bus] - v[t_bus]))
    end
    return im_vec
end

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
    return v_pert, Y_pert
end

@testset "AC PF Topology Sensitivities (:g, :b)" begin
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")

    @testset "Finite-difference verification — case5" begin
        file = joinpath(pm_path, "case5.m")
        pm_data = PowerModels.parse_file(file)
        net_data = PowerModels.make_basic_network(pm_data)
        PowerModels.compute_ac_pf!(net_data)
        state = ACPowerFlowState(net_data)

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
                    v_p, Y_p = _perturbed_voltage(state, param, e, ε, p_base, q_base)
                    v_m, Y_m = _perturbed_voltage(state, param, e, -ε, p_base, q_base)

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

                    fd_f = (_branch_flows(v_p, Y_p, state.branch_data) -
                            _branch_flows(v_m, Y_m, state.branch_data)) / (2ε)
                    if norm(fd_f) > 1e-6
                        rel_err_f = norm(Matrix(S_f)[:, e] - fd_f) / norm(fd_f)
                        @test rel_err_f < 1e-2
                    end

                    fd_im = (_branch_currents_mag(v_p, Y_p, state.branch_data) -
                             _branch_currents_mag(v_m, Y_m, state.branch_data)) / (2ε)
                    if norm(fd_im) > 1e-6
                        rel_err_im = norm(Matrix(S_im)[:, e] - fd_im) / norm(fd_im)
                        @test rel_err_im < 1e-2
                    end
                end
            end
        end
    end

    @testset "Smoke tests — all 10 combinations" begin
        file = joinpath(pm_path, "case5.m")
        pm_data = PowerModels.parse_file(file)
        net_data = PowerModels.make_basic_network(pm_data)
        PowerModels.compute_ac_pf!(net_data)
        state = ACPowerFlowState(net_data)

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
        pm_data = PowerModels.parse_file(file)
        net_data = PowerModels.make_basic_network(pm_data)
        PowerModels.compute_ac_pf!(net_data)
        full_state = ACPowerFlowState(net_data)

        raw_state = ACPowerFlowState(full_state.v, full_state.Y;
            idx_slack=full_state.idx_slack, branch_data=full_state.branch_data)
        @test isnothing(raw_state.net)
        @test_throws ArgumentError calc_sensitivity(raw_state, :vm, :g)
        @test_throws ArgumentError calc_sensitivity(raw_state, :va, :b)
    end

    @testset "Non-basic network — col_to_id maps to branch IDs" begin
        file = joinpath(pm_path, "case5.m")
        pm_data = PowerModels.parse_file(file)
        PowerModels.compute_ac_pf!(pm_data)
        state = ACPowerFlowState(pm_data)

        S = calc_sensitivity(state, :vm, :g)
        @test S.formulation == :acpf
        @test length(S.col_to_id) == state.m
        @test S.col_to_id == state.net.id_map.branch_ids
        @test all(isfinite, Matrix(S))
    end

    @testset "Sensitivity metadata" begin
        file = joinpath(pm_path, "case14.m")
        pm_data = PowerModels.parse_file(file)
        net_data = PowerModels.make_basic_network(pm_data)
        PowerModels.compute_ac_pf!(net_data)
        state = ACPowerFlowState(net_data)

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
