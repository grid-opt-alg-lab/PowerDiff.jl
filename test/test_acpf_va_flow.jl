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

# FD verification of AC PF voltage angle (∂θ/∂p, ∂θ/∂q) and branch active power
# flow (∂P_flow/∂p, ∂P_flow/∂q) sensitivities. Same methodology as
# test_ac_pf_verification.jl.
#
# Verifies :va and :f operands against finite-difference re-solve,
# using the same PQ-only Newton approach as test_ac_pf_verification.jl.

using PowerDiff
using PowerModels
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Test

# PQ Newton solver: pf_residual_pq / solve_pf_pq from common.jl
@isdefined(pf_residual_pq) || include("common.jl")

@testset "AC PF Voltage Angle & Branch Flow Verification" begin
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")

    pm_data = PowerModels.parse_file(file)
    net_data = PowerModels.make_basic_network(pm_data)

    pf_data = deepcopy(net_data)
    PowerModels.compute_ac_pf!(pf_data)

    v_base = PowerModels.calc_basic_bus_voltage(pf_data)
    Y = PowerModels.calc_basic_admittance_matrix(pf_data)
    pf_ref = PowerModels.build_ref(deepcopy(pf_data))[:it][:pm][:nw][0]
    slack = first(keys(pf_ref[:ref_buses]))
    n = length(v_base)
    n_branch = length(pf_data["branch"])

    state = ACPowerFlowState(v_base, Y;
        idx_slack=slack, branch_data=pf_data["branch"])

    non_slack = [i for i in 1:n if i != slack]

    I_base = Y * v_base
    S_base = v_base .* conj.(I_base)
    p_base = real.(S_base)[non_slack]
    q_base = imag.(S_base)[non_slack]

    # Compute base branch power flows
    pflow_base = zeros(n_branch)
    for (_, br) in pf_data["branch"]
        ℓ = br["index"]
        f = br["f_bus"]
        t = br["t_bus"]
        Y_ft = Y[f, t]
        I_ℓ = Y_ft * (v_base[f] - v_base[t])
        pflow_base[ℓ] = real(v_base[f] * conj(I_ℓ))
    end

    delta = 1e-5
    fd_tol = 0.01  # 1% relative error
    # Forward-difference parameters: delta=1e-5 gives O(1e-5) truncation error.
    # fd_tol=0.01 (1%); see test_ac_pf_verification.jl for the full error analysis.

    # Get analytical sensitivities
    dva_dp = calc_sensitivity(state, :va, :p)
    dva_dq = calc_sensitivity(state, :va, :q)
    df_dp = calc_sensitivity(state, :f, :p)
    df_dq = calc_sensitivity(state, :f, :q)

    # -----------------------------------------------------------------
    # ∂θ/∂p — perturb active power, check angle change
    # -----------------------------------------------------------------
    @testset "∂θ/∂p (voltage angle w.r.t. active power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            p_pert = copy(p_base)
            p_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_pert, q_base, slack)
            fd_dva = (angle.(v_new) - angle.(v_base)) / delta

            analytical_col = Matrix(dva_dp)[:, k_global]

            if norm(fd_dva) > 1e-10
                rel_err = norm(analytical_col - fd_dva) / norm(fd_dva)
                @test rel_err < fd_tol
            else
                @info "Skipped ∂θ/∂p FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂θ/∂q — perturb reactive power, check angle change
    # -----------------------------------------------------------------
    @testset "∂θ/∂q (voltage angle w.r.t. reactive power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            q_pert = copy(q_base)
            q_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_base, q_pert, slack)
            fd_dva = (angle.(v_new) - angle.(v_base)) / delta

            analytical_col = Matrix(dva_dq)[:, k_global]

            if norm(fd_dva) > 1e-10
                rel_err = norm(analytical_col - fd_dva) / norm(fd_dva)
                @test rel_err < fd_tol
            else
                @info "Skipped ∂θ/∂q FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂P_flow/∂p — perturb active power, check flow change
    # -----------------------------------------------------------------
    @testset "∂P_flow/∂p (branch flow w.r.t. active power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            p_pert = copy(p_base)
            p_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_pert, q_base, slack)

            pflow_pert = zeros(n_branch)
            for (_, br) in pf_data["branch"]
                ℓ = br["index"]
                f = br["f_bus"]
                t = br["t_bus"]
                Y_ft = Y[f, t]
                I_ℓ = Y_ft * (v_new[f] - v_new[t])
                pflow_pert[ℓ] = real(v_new[f] * conj(I_ℓ))
            end

            fd_df = (pflow_pert - pflow_base) / delta
            analytical_col = Matrix(df_dp)[:, k_global]

            active = abs.(pflow_base) .> 1e-6
            if norm(fd_df[active]) > 1e-10
                rel_err = norm(analytical_col[active] - fd_df[active]) / norm(fd_df[active])
                @test rel_err < fd_tol
            else
                @info "Skipped ∂P_flow/∂p FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂P_flow/∂q — perturb reactive power, check flow change
    # -----------------------------------------------------------------
    @testset "∂P_flow/∂q (branch flow w.r.t. reactive power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            q_pert = copy(q_base)
            q_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_base, q_pert, slack)

            pflow_pert = zeros(n_branch)
            for (_, br) in pf_data["branch"]
                ℓ = br["index"]
                f = br["f_bus"]
                t = br["t_bus"]
                Y_ft = Y[f, t]
                I_ℓ = Y_ft * (v_new[f] - v_new[t])
                pflow_pert[ℓ] = real(v_new[f] * conj(I_ℓ))
            end

            fd_df = (pflow_pert - pflow_base) / delta
            analytical_col = Matrix(df_dq)[:, k_global]

            active = abs.(pflow_base) .> 1e-6
            if norm(fd_df[active]) > 1e-10
                rel_err = norm(analytical_col[active] - fd_df[active]) / norm(fd_df[active])
                @test rel_err < fd_tol
            else
                @info "Skipped ∂P_flow/∂q FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end
end
