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

# FD verification of AC PF sensitivities (∂|V|/∂p, ∂|V|/∂q, ∂|I|/∂p, ∂|I|/∂q,
# ∂V/∂p, ∂V/∂q). Perturbs each injection, re-solves Newton-Raphson via
# `solve_pf_pq`, and compares against analytical sensitivities. Uses case5 with
# all non-slack buses treated as PQ.
#
# Key: the analytical formula treats ALL non-slack buses as PQ (free voltage),
# so we cannot use PowerModels' compute_ac_pf! which enforces PV constraints.
# Instead, we implement a PQ-only Newton solve and compare converged solutions.
#
# Important: we construct ACPowerFlowState using PowerModels' admittance matrix
# (Y_pm) to ensure consistency with the solved voltage operating point.

using PowerDiff
using PowerModels
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Test

# PQ Newton solver: pf_residual_pq / solve_pf_pq from common.jl
@isdefined(pf_residual_pq) || include("common.jl")

@testset "AC PF Finite-Difference Verification" begin
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")

    pm_data = PowerModels.parse_file(file)
    net_data = PowerModels.make_basic_network(pm_data)

    # Solve base AC power flow
    pf_data = deepcopy(net_data)
    PowerModels.compute_ac_pf!(pf_data)

    # Use PowerModels' admittance matrix for consistency with the PF solution.
    v_base = PowerModels.calc_basic_bus_voltage(pf_data)
    Y = PowerModels.calc_basic_admittance_matrix(pf_data)
    # Find slack bus from reference bus data
    pf_ref = PowerModels.build_ref(deepcopy(pf_data))[:it][:pm][:nw][0]
    slack = first(keys(pf_ref[:ref_buses]))
    n = length(v_base)
    n_branch = length(pf_data["branch"])

    # Construct state with the PM admittance matrix
    state = ACPowerFlowState(v_base, Y;
        idx_slack=slack, branch_data=pf_data["branch"])

    # Get analytical sensitivities
    dvm_dp = calc_sensitivity(state, :vm, :p)
    dvm_dq = calc_sensitivity(state, :vm, :q)
    dim_dp = calc_sensitivity(state, :im, :p)
    dim_dq = calc_sensitivity(state, :im, :q)

    non_slack = [i for i in 1:n if i != slack]

    # Base power injections using standard convention: S = V .* conj(I), I = Y*V
    I_base = Y * v_base
    S_base = v_base .* conj.(I_base)
    p_base = real.(S_base)[non_slack]
    q_base = imag.(S_base)[non_slack]

    # Compute base branch currents
    im_base = zeros(n_branch)
    for (_, br) in pf_data["branch"]
        ℓ = br["index"]
        f = br["f_bus"]
        t = br["t_bus"]
        I_ℓ = Y[f, t] * (v_base[f] - v_base[t])
        im_base[ℓ] = abs(I_ℓ)
    end

    delta = 1e-5
    fd_tol = 0.01  # 1% relative error
    # Forward-difference error model:
    #   Truncation error: O(delta) from Taylor remainder
    #   Cancellation error: O(eps_mach/delta) ≈ O(1e-11), negligible
    #   Newton re-solve converges to O(1e-12), so FD accuracy is O(delta) ≈ 1e-5
    #   fd_tol = 1% provides a generous margin; typical agreement is O(1e-4)

    # -----------------------------------------------------------------
    # ∂|v|/∂p — perturb active power at non-slack buses
    # -----------------------------------------------------------------
    @testset "∂|v|/∂p (voltage magnitude w.r.t. active power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            p_pert = copy(p_base)
            p_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_pert, q_base, slack)
            fd_dvm = (abs.(v_new) - abs.(v_base)) / delta

            analytical_col = Matrix(dvm_dp)[:, k_global]

            if norm(fd_dvm) > 1e-10
                rel_err = norm(analytical_col - fd_dvm) / norm(fd_dvm)
                @test rel_err < fd_tol
            else
                @info "Skipped ∂|v|/∂p FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂|v|/∂q — perturb reactive power at non-slack buses
    # -----------------------------------------------------------------
    @testset "∂|v|/∂q (voltage magnitude w.r.t. reactive power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            q_pert = copy(q_base)
            q_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_base, q_pert, slack)
            fd_dvm = (abs.(v_new) - abs.(v_base)) / delta

            analytical_col = Matrix(dvm_dq)[:, k_global]

            if norm(fd_dvm) > 1e-10
                rel_err = norm(analytical_col - fd_dvm) / norm(fd_dvm)
                @test rel_err < fd_tol
            else
                @info "Skipped ∂|v|/∂q FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂|I|/∂p — perturb active power, check current magnitude change
    # -----------------------------------------------------------------
    @testset "∂|I|/∂p (current magnitude w.r.t. active power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            p_pert = copy(p_base)
            p_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_pert, q_base, slack)

            im_pert = zeros(n_branch)
            for (_, br) in pf_data["branch"]
                ℓ = br["index"]
                f = br["f_bus"]
                t = br["t_bus"]
                I_ℓ = Y[f, t] * (v_new[f] - v_new[t])
                im_pert[ℓ] = abs(I_ℓ)
            end

            fd_dim = (im_pert - im_base) / delta
            analytical_col = Matrix(dim_dp)[:, k_global]

            active = im_base .> 1e-6
            if norm(fd_dim[active]) > 1e-10
                rel_err = norm(analytical_col[active] - fd_dim[active]) / norm(fd_dim[active])
                @test rel_err < fd_tol
            else
                @info "Skipped ∂|I|/∂p FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂|I|/∂q — perturb reactive power, check current magnitude change
    # -----------------------------------------------------------------
    @testset "∂|I|/∂q (current magnitude w.r.t. reactive power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            q_pert = copy(q_base)
            q_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_base, q_pert, slack)

            im_pert = zeros(n_branch)
            for (_, br) in pf_data["branch"]
                ℓ = br["index"]
                f = br["f_bus"]
                t = br["t_bus"]
                I_ℓ = Y[f, t] * (v_new[f] - v_new[t])
                im_pert[ℓ] = abs(I_ℓ)
            end

            fd_dim = (im_pert - im_base) / delta
            analytical_col = Matrix(dim_dq)[:, k_global]

            active = im_base .> 1e-6
            if norm(fd_dim[active]) > 1e-10
                rel_err = norm(analytical_col[active] - fd_dim[active]) / norm(fd_dim[active])
                @test rel_err < fd_tol
            else
                @info "Skipped ∂|I|/∂q FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂v/∂p — complex phasor sensitivity w.r.t. active power
    # This test would have caught the conjugate sign bug in ∂v reconstruction
    # -----------------------------------------------------------------
    dv_dp = calc_sensitivity(state, :v, :p)
    dv_dq = calc_sensitivity(state, :v, :q)

    @testset "∂v/∂p (complex phasor w.r.t. active power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            p_pert = copy(p_base)
            p_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_pert, q_base, slack)
            fd_dv = (v_new - v_base) / delta

            analytical_col = Matrix(dv_dp)[:, k_global]

            if norm(fd_dv) > 1e-10
                rel_err = norm(analytical_col - fd_dv) / norm(fd_dv)
                @test rel_err < fd_tol
            else
                @info "Skipped ∂v/∂p FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end

    # -----------------------------------------------------------------
    # ∂v/∂q — complex phasor sensitivity w.r.t. reactive power
    # -----------------------------------------------------------------
    @testset "∂v/∂q (complex phasor w.r.t. reactive power)" begin
        for k_local in 1:min(3, length(non_slack))
            k_global = non_slack[k_local]

            q_pert = copy(q_base)
            q_pert[k_local] += delta

            v_new = solve_pf_pq(Y, v_base, p_base, q_pert, slack)
            fd_dv = (v_new - v_base) / delta

            analytical_col = Matrix(dv_dq)[:, k_global]

            if norm(fd_dv) > 1e-10
                rel_err = norm(analytical_col - fd_dv) / norm(fd_dv)
                @test rel_err < fd_tol
            else
                @info "Skipped ∂v/∂q FD: near-zero perturbation" bus=k_global
                @test norm(analytical_col) < 1e-6  # analytical must also be near-zero
            end
        end
    end
end
