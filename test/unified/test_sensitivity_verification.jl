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

# Cross-verification of DC PF sensitivities against ForwardDiff (exact) and
# DC OPF sensitivities against finite differences (demand, switching).
# ForwardDiff tests differentiate through the same linear algebra as the
# analytical code, so they verify correctness rather than just consistency.

using PowerDiff
using PowerModels
using ForwardDiff
using LinearAlgebra
using Test

@testset "Sensitivity Verification with ForwardDiff" begin

    # Load a test network
    case_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower", "case5.m")
    data = PowerModels.parse_file(case_path)
    net_data = PowerModels.make_basic_network(data)

    @testset "DC Power Flow Switching Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Solve DC power flow
        pf_state = DCPowerFlowState(net, d)

        # Get analytical sensitivity via new API
        dva_dsw = calc_sensitivity(pf_state, :va, :sw)
        df_dsw = calc_sensitivity(pf_state, :f, :sw)

        # Define a function that computes theta as a function of sw
        # Uses reduced Laplacian (ref bus deleted) to match the implementation.
        function theta_of_sw(sw_vec)
            A = net.A
            b = net.b
            L = transpose(A) * Diagonal(-b .* sw_vec) * A
            non_ref = setdiff(1:net.n, net.ref_bus)
            L_r = Matrix(L[non_ref, non_ref])

            θ = zeros(eltype(sw_vec), net.n)
            θ[non_ref] = L_r \ pf_state.p[non_ref]
            return θ
        end

        # Compute ForwardDiff Jacobian
        sw0 = copy(net.sw)
        fd_dva_dsw = ForwardDiff.jacobian(theta_of_sw, sw0)

        # Both paths compute the same linear algebra (L_r \ p_r), so agreement
        # should be O(eps_mach * cond(B_r)) — well within 1e-10.
        @test size(dva_dsw) == size(fd_dva_dsw)
        @test maximum(abs.(Matrix(dva_dsw) - fd_dva_dsw)) < 1e-10

        # Also verify flow sensitivity
        function flow_of_sw(sw_vec)
            A = net.A
            b = net.b
            θ = theta_of_sw(sw_vec)
            W = Diagonal(-b .* sw_vec)
            return W * A * θ
        end

        fd_df_dsw = ForwardDiff.jacobian(flow_of_sw, sw0)
        @test size(df_dsw) == size(fd_df_dsw)
        @test maximum(abs.(Matrix(df_dsw) - fd_df_dsw)) < 1e-10
    end

    @testset "DC Power Flow Demand Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Solve DC power flow
        pf_state = DCPowerFlowState(net, d)

        # Get analytical sensitivity via new API
        dva_dd = calc_sensitivity(pf_state, :va, :d)

        # Define a function that computes theta as a function of d
        # Uses reduced Laplacian (ref bus deleted) to match the implementation.
        function theta_of_d(d_vec)
            A = net.A
            b = net.b
            L = transpose(A) * Diagonal(-b .* net.sw) * A
            non_ref = setdiff(1:net.n, net.ref_bus)
            L_r = Matrix(L[non_ref, non_ref])
            p = pf_state.pg - d_vec  # Net injection
            θ = zeros(eltype(d_vec), net.n)
            θ[non_ref] = L_r \ p[non_ref]
            return θ
        end

        # Compute ForwardDiff Jacobian
        fd_dva_dd = ForwardDiff.jacobian(theta_of_d, d)

        # Compare analytical vs ForwardDiff
        @test size(dva_dd) == size(fd_dva_dd)
        @test maximum(abs.(Matrix(dva_dd) - fd_dva_dd)) < 1e-8
    end

    @testset "DC Power Flow PTDF (∂f/∂d)" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Solve DC power flow
        pf_state = DCPowerFlowState(net, d)

        # Get analytical PTDF via API
        df_dd = calc_sensitivity(pf_state, :f, :d)

        # Define flow as function of demand: f(d) = W · A · L_r⁻¹ · (g - d)
        # Uses reduced Laplacian (ref bus deleted) to match the implementation.
        function flow_of_d(d_vec)
            A = net.A
            b = net.b
            L = transpose(A) * Diagonal(-b .* net.sw) * A
            non_ref = setdiff(1:net.n, net.ref_bus)
            L_r = Matrix(L[non_ref, non_ref])
            p = pf_state.pg - d_vec  # Net injection
            θ = zeros(eltype(d_vec), net.n)
            θ[non_ref] = L_r \ p[non_ref]
            W = Diagonal(-b .* net.sw)
            return W * A * θ
        end

        # Compute ForwardDiff Jacobian
        fd_df_dd = ForwardDiff.jacobian(flow_of_d, d)

        # Compare analytical vs ForwardDiff
        @test size(df_dd) == size(fd_df_dd)
        @test maximum(abs.(Matrix(df_dd) - fd_df_dd)) < 1e-8
    end

    @testset "DC OPF Demand Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Get analytical sensitivity via new API
        dva_dd = calc_sensitivity(prob, :va, :d)

        # Verify with finite differences
        # 1e-3 absolute tolerance: demand perturbation doesn't change the active set,
        # so FD accuracy is O(ε) ≈ 1e-5 plus Ipopt noise O(tol/ε) ≈ 1e-3.
        ε = 1e-5
        n = net.n

        for i in 1:min(3, n)
            d_pert = copy(d)
            d_pert[i] += ε

            update_demand!(prob, d_pert)
            sol_pert = solve!(prob)

            fd_dva_dd_col = (sol_pert.va - sol.va) / ε

            max_err = maximum(abs.(Matrix(dva_dd)[:, i] - fd_dva_dd_col))
            @test max_err < 1e-3

            update_demand!(prob, d)
        end
    end

    @testset "DC OPF Switching Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Get analytical sensitivity via new API
        dva_dsw = calc_sensitivity(prob, :va, :sw)

        # Verify with finite differences (negative perturbation to stay in [0,1])
        ε = 1e-5
        m = net.m

        for e in 1:min(3, m)
            sw_pert = copy(net.sw)
            sw_pert[e] -= ε

            net_pert = DCNetwork(net_data)
            net_pert.sw .= sw_pert
            prob_pert = DCOPFProblem(net_pert, d)
            sol_pert = solve!(prob_pert)

            fd_dva_dsw_col = (sol.va - sol_pert.va) / ε  # Reversed due to negative ε

            # 5% tolerance: switching perturbation can shift constraints near their
            # binding threshold, causing discontinuous dual changes that amplify FD error.
            max_err = maximum(abs.(Matrix(dva_dsw)[:, e] - fd_dva_dsw_col))
            @test max_err < 0.05
        end
    end

    @testset "DC OPF Switching Sensitivity (∂pg/∂sw, ∂f/∂sw)" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Get analytical sensitivities via API
        dpg_dsw = calc_sensitivity(prob, :pg, :sw)
        df_dsw = calc_sensitivity(prob, :f, :sw)

        # Verify with finite differences (negative perturbation to stay in [0,1])
        ε = 1e-5
        m = net.m

        for e in 1:min(3, m)
            sw_pert = copy(net.sw)
            sw_pert[e] -= ε

            net_pert = DCNetwork(net_data)
            net_pert.sw .= sw_pert
            prob_pert = DCOPFProblem(net_pert, d)
            sol_pert = solve!(prob_pert)

            # Reversed sign due to negative perturbation
            fd_dpg_col = (sol.pg - sol_pert.pg) / ε
            fd_df_col = (sol.f - sol_pert.f) / ε

            # Check ∂pg/∂sw
            if norm(fd_dpg_col) > 1e-10
                rel_err_pg = norm(Matrix(dpg_dsw)[:, e] - fd_dpg_col) / norm(fd_dpg_col)
                @test rel_err_pg < 0.05
            else
                @info "Skipped ∂pg/∂sw FD: near-zero perturbation" branch=e
                @test norm(Matrix(dpg_dsw)[:, e]) < 1e-6
            end

            # Check ∂f/∂sw
            if norm(fd_df_col) > 1e-10
                rel_err_f = norm(Matrix(df_dsw)[:, e] - fd_df_col) / norm(fd_df_col)
                @test rel_err_f < 0.05
            else
                @info "Skipped ∂f/∂sw FD: near-zero perturbation" branch=e
                @test norm(Matrix(df_dsw)[:, e]) < 1e-6
            end
        end
    end

    @testset "AC Voltage-Power Sensitivity" begin
        # Solve AC power flow
        PowerModels.compute_ac_pf!(net_data)
        state = ACPowerFlowState(net_data)

        # Get analytical sensitivity via new API
        dvm_dp = calc_sensitivity(state, :vm, :p)
        dvm_dq = calc_sensitivity(state, :vm, :q)

        # Verify structure
        @test size(dvm_dp) == (state.n, state.n)
        @test size(dvm_dq) == (state.n, state.n)

        # Basic sanity: sensitivities should be real and finite
        @test all(isfinite, Matrix(dvm_dp))
        @test all(isfinite, Matrix(dvm_dq))

        # Slack bus voltage should have zero sensitivity
        slack_idx = state.idx_slack
        @test maximum(abs.(Matrix(dvm_dp)[slack_idx, :])) < 1e-10
        @test maximum(abs.(Matrix(dvm_dq)[slack_idx, :])) < 1e-10
    end

end
