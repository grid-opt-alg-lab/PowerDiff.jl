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

# Load-shedding (:psh) operand tests

# Local helper for the 2-bus topology used by multiple testsets
function _make_2bus_psh(; gmax=0.5, cq=0.0, cl=10.0, tau=0.0)
    n, m, k = 2, 1, 1
    A = sparse([1.0 -1.0])
    G_inc = sparse(reshape([1.0, 0.0], 2, 1))
    b = [-10.0]
    DCNetwork(n, m, k, A, G_inc, b;
        fmax=[100.0], gmax=[gmax], gmin=[0.0],
        cl=[cl], cq=[cq], c_shed=[1e4, 1e4],
        ref_bus=1, tau=tau)
end

# Helper: load case14 with tightened flow limits so congestion forces shedding.
#
# With original generator capacities (total 7.724 >> demand 2.59), tight flow limits
# prevent power delivery to some buses, causing congestion-induced shedding while
# keeping most generators interior (not at bounds). This avoids the degenerate
# complementarity that occurs when all generators are at their upper bounds
# (singular KKT matrix from zero diagonals in ρ_ub rows).
#
# fmax_scale=0.03 produces ~0.05 MW of shedding with 4 interior generators.
function _make_case14_shedding(; fmax_scale=0.03)
    net_data = load_test_case("case14.m")
    isnothing(net_data) && return nothing
    dc_net = DCNetwork(net_data)
    d = calc_demand_vector(net_data)
    dc_net.fmax .*= fmax_scale
    return (; net_data, dc_net, d, fmax_scale)
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
            residual = dc_net.G_inc * sol.pg + sol.psh - d - B_mat * sol.va
            @test norm(residual) < 1e-4
        end
    end

    @testset "psh > 0: insufficient generation (2-bus)" begin
        dc_net = _make_2bus_psh(gmax=0.5)

        d = [0.0, 1.0]  # 1 MW demand at bus 2, but gmax = 0.5
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        # Generator at max, shedding makes up the difference
        @test sol.pg[1] ≈ 0.5 atol=1e-4
        @test sum(sol.psh) ≈ 0.5 atol=1e-4

        # Power balance: G_inc * g + psh - d ≈ B * θ
        B_mat = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        residual = dc_net.G_inc * sol.pg + sol.psh - d - B_mat * sol.va
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
            ref_bus=1, tau=0.01)

        d = [0.0, 0.0, 1.0]  # 1 MW load at bus 3
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        # Some shedding should occur at bus 3 because flow limits prevent full delivery
        @test sum(sol.psh) > 0.1

        # Power balance still holds
        B_mat = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        residual = dc_net.G_inc * sol.pg + sol.psh - d - B_mat * sol.va
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
        psh_formula = d + B_mat * sol.va - dc_net.G_inc * sol.pg
        @test isapprox(sol.psh, psh_formula, atol=1e-4)
    end

    @testset "KKT residuals" begin
        # Test with active shedding
        dc_net = _make_2bus_psh(gmax=0.5, cq=1.0, tau=0.01)

        d = [0.0, 1.0]
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d)

        # Check primal feasibility (should be very tight)
        idx = kkt_indices(dc_net)
        @test norm(K[idx.nu_bal]) < 1e-4
        @test norm(K[idx.nu_flow]) < 1e-4

        # Also test with inactive shedding (feasible case)
        dc_net2 = _make_2bus_psh(gmax=10.0, cq=1.0, tau=0.01)

        d2 = [0.0, 1.0]
        prob2 = DCOPFProblem(dc_net2, d2)
        sol2 = solve!(prob2)

        z2 = flatten_variables(sol2, prob2)
        K2 = kkt(z2, prob2, d2)
        idx2 = kkt_indices(dc_net2)
        @test norm(K2[idx2.nu_bal]) < 1e-4
        @test norm(K2[idx2.nu_flow]) < 1e-4
    end

    # =========================================================================
    # Finite-Difference Verification on case14
    #
    # Uses congestion-based shedding (tight fmax) on case14. This keeps most
    # generators interior (not at bounds), ensuring a well-conditioned KKT
    # Jacobian. Generation-insufficiency scenarios push all generators to their
    # upper bounds, creating singular KKT matrices that amplify errors.
    # =========================================================================

    @testset "FD verification: ∂psh/∂d (case14)" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; dc_net, d) = setup

            prob = DCOPFProblem(dc_net, d)
            sol_base = solve!(prob)
            @test sum(sol_base.psh) > 1e-3  # Confirm shedding is active

            dpsh_dd = calc_sensitivity(prob, :psh, :d)

            # Finite difference — skip buses with d=0 (psh bounds collapse to [0,0]).
            delta = 1e-5
            for bus_idx in 1:dc_net.n
                d[bus_idx] == 0.0 && continue

                d_pert = copy(d)
                d_pert[bus_idx] += delta

                prob_pert = DCOPFProblem(dc_net, d_pert)
                sol_pert = solve!(prob_pert)

                fd = (sol_pert.psh - sol_base.psh) / delta
                analytical_col = Matrix(dpsh_dd)[:, bus_idx]

                if norm(fd) > 1e-6
                    rel_err = norm(analytical_col - fd) / norm(fd)
                    @test rel_err < 0.01
                else
                    @test norm(analytical_col) < 1e-4
                end
            end
        end
    end

    @testset "FD verification: ∂psh/∂sw (case14)" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; net_data, dc_net, d, fmax_scale) = setup

            prob = DCOPFProblem(dc_net, d)
            sol_base = solve!(prob)
            @test sum(sol_base.psh) > 1e-3  # Confirm shedding is active

            dpsh_dsw = calc_sensitivity(prob, :psh, :sw)

            delta = 1e-5
            for branch_idx in 1:dc_net.m
                dc_net_pert = DCNetwork(net_data)
                dc_net_pert.fmax .*= fmax_scale
                dc_net_pert.sw[branch_idx] += delta

                prob_pert = DCOPFProblem(dc_net_pert, d)
                sol_pert = solve!(prob_pert)

                fd = (sol_pert.psh - sol_base.psh) / delta
                analytical_col = Matrix(dpsh_dsw)[:, branch_idx]

                if norm(fd) > 1e-4
                    rel_err = norm(analytical_col - fd) / norm(fd)
                    @test rel_err < 0.05
                else
                    @test norm(analytical_col) < 1e-2
                end
            end
        end
    end

    @testset "Conservation: sum(∂pg/∂sw) + sum(∂psh/∂sw) = 0" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; dc_net, d) = setup

            prob = DCOPFProblem(dc_net, d)
            solve!(prob)

            dpg_dsw = Matrix(calc_sensitivity(prob, :pg, :sw))
            dpsh_dsw = Matrix(calc_sensitivity(prob, :psh, :sw))

            # From power balance: sum(g) + sum(psh) = sum(d).
            # Since d doesn't depend on sw: sum(∂pg/∂sw_j) + sum(∂psh/∂sw_j) = 0.
            for j in 1:dc_net.m
                @test abs(sum(dpg_dsw[:, j]) + sum(dpsh_dsw[:, j])) < 1e-4
            end
        end
    end

    @testset "FD verification: ∂psh/∂cq (case14)" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; net_data, dc_net, d, fmax_scale) = setup

            prob = DCOPFProblem(dc_net, d)
            sol_base = solve!(prob)
            @test sum(sol_base.psh) > 1e-3

            dpsh_dcq = calc_sensitivity(prob, :psh, :cq)

            delta = 1e-5
            for gen_idx in 1:dc_net.k
                dc_net_pert = DCNetwork(net_data)
                dc_net_pert.fmax .*= fmax_scale
                dc_net_pert.cq[gen_idx] += delta

                prob_pert = DCOPFProblem(dc_net_pert, d)
                sol_pert = solve!(prob_pert)

                fd = (sol_pert.psh - sol_base.psh) / delta
                analytical_col = Matrix(dpsh_dcq)[:, gen_idx]

                if norm(fd) > 1e-6
                    rel_err = norm(analytical_col - fd) / norm(fd)
                    @test rel_err < 0.05
                else
                    @test norm(analytical_col) < 1e-4
                end
            end
        end
    end

    @testset "FD verification: ∂psh/∂cl (case14)" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; net_data, dc_net, d, fmax_scale) = setup

            prob = DCOPFProblem(dc_net, d)
            sol_base = solve!(prob)
            @test sum(sol_base.psh) > 1e-3

            dpsh_dcl = calc_sensitivity(prob, :psh, :cl)

            delta = 1e-5
            for gen_idx in 1:dc_net.k
                dc_net_pert = DCNetwork(net_data)
                dc_net_pert.fmax .*= fmax_scale
                dc_net_pert.cl[gen_idx] += delta

                prob_pert = DCOPFProblem(dc_net_pert, d)
                sol_pert = solve!(prob_pert)

                fd = (sol_pert.psh - sol_base.psh) / delta
                analytical_col = Matrix(dpsh_dcl)[:, gen_idx]

                if norm(fd) > 1e-6
                    rel_err = norm(analytical_col - fd) / norm(fd)
                    @test rel_err < 0.05
                else
                    @test norm(analytical_col) < 1e-4
                end
            end
        end
    end

    @testset "FD verification: ∂psh/∂fmax (case14)" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; net_data, dc_net, d, fmax_scale) = setup

            prob = DCOPFProblem(dc_net, d)
            sol_base = solve!(prob)
            @test sum(sol_base.psh) > 1e-3

            dpsh_dfmax = calc_sensitivity(prob, :psh, :fmax)

            delta = 1e-5
            for branch_idx in 1:dc_net.m
                dc_net_pert = DCNetwork(net_data)
                dc_net_pert.fmax .*= fmax_scale
                dc_net_pert.fmax[branch_idx] += delta

                prob_pert = DCOPFProblem(dc_net_pert, d)
                sol_pert = solve!(prob_pert)

                fd = (sol_pert.psh - sol_base.psh) / delta
                analytical_col = Matrix(dpsh_dfmax)[:, branch_idx]

                if norm(fd) > 1e-6
                    rel_err = norm(analytical_col - fd) / norm(fd)
                    @test rel_err < 0.05
                else
                    @test norm(analytical_col) < 1e-4
                end
            end
        end
    end

    @testset "FD verification: ∂psh/∂b (case14)" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; net_data, dc_net, d, fmax_scale) = setup

            prob = DCOPFProblem(dc_net, d)
            sol_base = solve!(prob)
            @test sum(sol_base.psh) > 1e-3

            dpsh_db = calc_sensitivity(prob, :psh, :b)

            delta = 1e-5
            for branch_idx in 1:dc_net.m
                dc_net_pert = DCNetwork(net_data)
                dc_net_pert.fmax .*= fmax_scale
                dc_net_pert.b[branch_idx] += delta

                prob_pert = DCOPFProblem(dc_net_pert, d)
                sol_pert = solve!(prob_pert)

                fd = (sol_pert.psh - sol_base.psh) / delta
                analytical_col = Matrix(dpsh_db)[:, branch_idx]

                if norm(fd) > 1e-6
                    rel_err = norm(analytical_col - fd) / norm(fd)
                    @test rel_err < 0.05
                else
                    @test norm(analytical_col) < 1e-4
                end
            end
        end
    end

    @testset "Sensitivity types and dimensions" begin
        setup = _make_case14_shedding()
        if isnothing(setup)
            @test_skip false
        else
            (; dc_net, d) = setup
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

    @testset "Negative demand warning" begin
        dc_net = _make_2bus_psh(gmax=10.0)
        @test_warn "Negative demand" DCOPFProblem(dc_net, [0.0, -0.5])

        prob = DCOPFProblem(dc_net, [0.0, 1.0])
        @test_warn "Negative demand" update_demand!(prob, [0.0, -0.5])
    end

    @testset "Degenerate complementarity (d=0 everywhere)" begin
        # When d=0 at all buses, both shedding bounds collapse to 0 ≤ psh ≤ 0.
        # This triggers Tikhonov regularization in the KKT factorization.
        dc_net = _make_2bus_psh(gmax=10.0, cq=1.0, tau=0.01)
        d_zero = [0.0, 0.0]
        prob = DCOPFProblem(dc_net, d_zero)
        sol = solve!(prob)

        # No load → no shedding
        @test all(abs.(sol.psh) .< 1e-6)

        # All psh sensitivities should be finite (regularization keeps KKT invertible)
        for param in [:d, :sw, :cq, :cl, :fmax, :b]
            S = calc_sensitivity(prob, :psh, param)
            @test all(isfinite, Matrix(S))
        end

        # Other operands should also produce finite sensitivities
        for op in [:va, :pg, :f, :lmp]
            S = calc_sensitivity(prob, op, :d)
            @test all(isfinite, Matrix(S))
        end
    end

end
