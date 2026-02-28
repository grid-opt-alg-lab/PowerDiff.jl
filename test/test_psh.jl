# Load-shedding (:psh) operand tests

# Local helper for the 2-bus topology used by multiple testsets
function _make_2bus_psh(; gmax=0.5, cq=0.0, cl=10.0, τ=0.0)
    n, m, k = 2, 1, 1
    A = sparse([1.0 -1.0])
    G_inc = sparse(reshape([1.0, 0.0], 2, 1))
    b = [-10.0]
    DCNetwork(n, m, k, A, G_inc, b;
        fmax=[100.0], gmax=[gmax], gmin=[0.0],
        cl=[cl], cq=[cq], c_shed=[1e4, 1e4],
        ref_bus=1, τ=τ)
end

@testset "Load Shedding (psh)" begin

    @testset "psh ≈ 0 when feasible (case5)" begin
        net_data = load_test_case("case5.m")
        if isnothing(net_data)
            @test_skip false
        else
            dc_net = DCNetwork(net_data)
            d = calc_demand_vector(net_data)
            prob = DCOPFProblem(dc_net, d)
            sol = solve!(prob)

            # With normal demand and sufficient generation, no shedding needed
            @test all(abs.(sol.psh) .< 1e-4)

            # Power balance still holds: G_inc * g + psh - d ≈ B * θ
            B_mat = PowerModelsDiff.calc_susceptance_matrix(dc_net)
            residual = dc_net.G_inc * sol.g + sol.psh - d - B_mat * sol.θ
            @test norm(residual) < 1e-4
        end
    end

    @testset "psh > 0: insufficient generation (2-bus)" begin
        dc_net = _make_2bus_psh(gmax=0.5)

        d = [0.0, 1.0]  # 1 MW demand at bus 2, but gmax = 0.5
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        # Generator at max, shedding makes up the difference
        @test sol.g[1] ≈ 0.5 atol=1e-4
        @test sum(sol.psh) ≈ 0.5 atol=1e-4

        # Power balance: G_inc * g + psh - d ≈ B * θ
        B_mat = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        residual = dc_net.G_inc * sol.g + sol.psh - d - B_mat * sol.θ
        @test norm(residual) < 1e-4
    end

    @testset "psh > 0: congestion (3-bus)" begin
        # Bus 1: cheap gen, Bus 2: no gen, Bus 3: load
        # Line 1→3 congested at 0.3 MW, line 2→3 doesn't help (no gen at bus 2)
        n, m, k = 3, 2, 1
        A = sparse([
            1.0  0.0 -1.0;   # Line 1→3
            0.0  1.0 -1.0    # Line 2→3
        ])
        G_inc = sparse(reshape([1.0, 0.0, 0.0], 3, 1))
        b = [-10.0, -10.0]

        dc_net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=[0.3, 0.3],  # Tight flow limits
            gmax=[10.0], gmin=[0.0],
            cl=[10.0], cq=[0.0], c_shed=[1e4, 1e4, 1e4],
            ref_bus=1, τ=0.01)

        d = [0.0, 0.0, 1.0]  # 1 MW load at bus 3
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        # Some shedding should occur at bus 3 because flow limits prevent full delivery
        @test sum(sol.psh) > 0.1

        # Power balance still holds
        B_mat = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        residual = dc_net.G_inc * sol.g + sol.psh - d - B_mat * sol.θ
        @test norm(residual) < 1e-4
    end

    @testset "psh satisfies power balance rearrangement" begin
        # Rearranging the power balance G_inc*g + psh - d = B*θ
        # gives psh = d + B*θ - G_inc*g.
        dc_net = _make_2bus_psh(gmax=0.7)

        d = [0.0, 1.0]
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        B_mat = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        psh_formula = d + B_mat * sol.θ - dc_net.G_inc * sol.g
        @test isapprox(sol.psh, psh_formula, atol=1e-4)
    end

    @testset "KKT residuals" begin
        # Test with active shedding
        dc_net = _make_2bus_psh(gmax=0.5, cq=1.0, τ=0.01)

        d = [0.0, 1.0]
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d)

        # Check primal feasibility (should be very tight)
        idx = kkt_indices(dc_net)
        @test norm(K[idx.ν_bal]) < 1e-4
        @test norm(K[idx.ν_flow]) < 1e-4

        # Also test with inactive shedding (feasible case)
        dc_net2 = _make_2bus_psh(gmax=10.0, cq=1.0, τ=0.01)

        d2 = [0.0, 1.0]
        prob2 = DCOPFProblem(dc_net2, d2)
        sol2 = solve!(prob2)

        z2 = flatten_variables(sol2, prob2)
        K2 = kkt(z2, prob2, d2)
        idx2 = kkt_indices(dc_net2)
        @test norm(K2[idx2.ν_bal]) < 1e-4
        @test norm(K2[idx2.ν_flow]) < 1e-4
    end

    @testset "FD verification: ∂psh/∂d" begin
        dc_net = _make_2bus_psh(gmax=0.5, cq=1.0, τ=0.01)
        n = dc_net.n

        d = [0.0, 1.0]
        prob = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob)

        dpsh_dd = calc_sensitivity(prob, :psh, :d)

        # Finite difference — verify each column where the active set is stable.
        # Skip buses with d=0: both psh bounds collapse to 0 ≤ psh ≤ 0,
        # making the active set change discontinuously under perturbation.
        delta = 1e-5
        for bus_idx in 1:n
            d[bus_idx] == 0.0 && continue

            d_pert = copy(d)
            d_pert[bus_idx] += delta

            prob_pert = DCOPFProblem(dc_net, d_pert)
            sol_pert = solve!(prob_pert)

            dpsh_dd_fd = (sol_pert.psh - sol_base.psh) / delta

            if norm(dpsh_dd_fd) > 1e-6
                rel_error = norm(Matrix(dpsh_dd)[:, bus_idx] - dpsh_dd_fd) / norm(dpsh_dd_fd)
                @test rel_error < 0.01
            else
                @test norm(Matrix(dpsh_dd)[:, bus_idx]) < 1e-4
            end
        end
    end

    @testset "FD verification: ∂psh/∂sw" begin
        # Use a 3-bus congested case where shedding is active
        n, m, k = 3, 2, 1
        A = sparse([
            1.0  0.0 -1.0;
            0.0  1.0 -1.0
        ])
        G_inc = sparse(reshape([1.0, 0.0, 0.0], 3, 1))
        b = [-10.0, -10.0]

        dc_net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=[0.3, 0.3],
            gmax=[10.0], gmin=[0.0],
            cl=[10.0], cq=[1.0], c_shed=[1e4, 1e4, 1e4],
            ref_bus=1, τ=0.01)

        d = [0.0, 0.0, 1.0]
        prob = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob)

        dpsh_dsw = calc_sensitivity(prob, :psh, :sw)

        # Finite difference on switching state of branch 1
        delta = 1e-5
        branch_idx = 1
        sw_pert = copy(dc_net.sw)
        sw_pert[branch_idx] += delta

        dc_net_pert = DCNetwork(n, m, k, A, G_inc, b;
            sw=sw_pert,
            fmax=[0.3, 0.3],
            gmax=[10.0], gmin=[0.0],
            cl=[10.0], cq=[1.0], c_shed=[1e4, 1e4, 1e4],
            ref_bus=1, τ=0.01)

        prob_pert = DCOPFProblem(dc_net_pert, d)
        sol_pert = solve!(prob_pert)

        dpsh_dsw_fd = (sol_pert.psh - sol_base.psh) / delta

        # Use absolute error for near-zero sensitivities (FD noise can dominate)
        if norm(dpsh_dsw_fd) > 1e-4
            rel_error = norm(Matrix(dpsh_dsw)[:, branch_idx] - dpsh_dsw_fd) / norm(dpsh_dsw_fd)
            @test rel_error < 0.1
        else
            @test norm(Matrix(dpsh_dsw)[:, branch_idx]) < 1e-2
        end
    end

    @testset "Sensitivity types and dimensions" begin
        net_data = load_test_case("case5.m")
        if isnothing(net_data)
            @test_skip false
        else
            dc_net = DCNetwork(net_data)
            d = calc_demand_vector(net_data)
            prob = DCOPFProblem(dc_net, d)
            solve!(prob)

            for param in [:d, :sw, :cq, :cl, :fmax, :b]
                S = calc_sensitivity(prob, :psh, param)
                @test S isa Sensitivity
                @test S.formulation == :dcopf
                @test S.operand == :psh
                @test S.parameter == param
                @test size(S, 1) == dc_net.n
                @test all(isfinite, Matrix(S))
            end
        end
    end

end
