# Phase angle difference dual tests for DC OPF KKT system (Issue #13)
#
# Verifies that gamma_lb/gamma_ub duals are correctly extracted and that
# the KKT system produces correct sensitivities when angle limits bind.

using PowerModelsDiff
using LinearAlgebra
using SparseArrays
using Test

@testset "Phase Angle Difference Duals" begin

    # 3-bus network: 1→2, 1→3, 2→3
    # Gen at bus 1 (cheap) and bus 2 (expensive), load distributed
    n, m, k = 3, 3, 2
    A = sparse([
        1.0  -1.0   0.0;   # Branch 1: 1→2
        1.0   0.0  -1.0;   # Branch 2: 1→3 (angle-constrained in tight tests)
        0.0   1.0  -1.0    # Branch 3: 2→3
    ])
    G_inc = sparse([
        1.0  0.0;   # Gen 1 at bus 1 (cheap)
        0.0  1.0;   # Gen 2 at bus 2 (expensive)
        0.0  0.0    # No gen at bus 3
    ])
    b = [-10.0, -10.0, -10.0]

    # Nonzero demand at all buses, small enough to avoid shedding
    d = [0.05, 0.05, 0.5]

    @testset "Loose angle limits (regression)" begin
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=fill(π, m), angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Gamma duals should be ~zero with loose limits
        @test maximum(abs.(sol.gamma_lb)) < 1e-4
        @test maximum(abs.(sol.gamma_ub)) < 1e-4

        # KKT residual: gamma-specific blocks should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d)
        idx = kkt_indices(net)
        @test norm(K[idx.gamma_lb]) < 1e-4
        @test norm(K[idx.gamma_ub]) < 1e-4
        @test norm(K[idx.nu_bal]) < 1e-4
        @test norm(K) < 1e-4  # full KKT residual

        # Sensitivities should be finite
        for (op, param) in [(:va, :d), (:pg, :d), (:f, :d), (:lmp, :d),
                            (:va, :sw), (:pg, :sw)]
            S = calc_sensitivity(prob, op, param)
            @test all(isfinite, Matrix(S))
        end
    end

    @testset "Tight angle limits — binding duals" begin
        # angmax = 0.025 on branch 2 (1→3): constrains angle difference
        angmax_tight = [π, 0.025, π]
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=angmax_tight, angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Branch 2 angle difference should be at or near the limit
        Aθ = net.A * sol.va
        @test Aθ[2] <= angmax_tight[2] + 1e-4

        # No shedding (sufficient gen capacity)
        @test maximum(abs.(sol.psh)) < 1e-6

        # Gamma_ub on branch 2 should be nonzero (binding upper angle limit)
        @test abs(sol.gamma_ub[2]) > 1e-2

        # Gamma duals should be non-negative (standard KKT convention)
        @test all(sol.gamma_lb .>= -1e-6)
        @test all(sol.gamma_ub .>= -1e-6)

        # Gamma-related KKT blocks should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d)
        idx = kkt_indices(net)
        @test norm(K[idx.gamma_lb]) < 1e-3
        @test norm(K[idx.gamma_ub]) < 1e-3
        @test norm(K[idx.nu_bal]) < 1e-4
        @test norm(K) < 1e-3  # full KKT residual
    end

    @testset "Tight lower angle limits — binding angmin" begin
        # Reverse the flow direction: load at bus 1, gen at bus 2 and bus 3
        # This makes A*θ negative on some branches, approaching angmin
        G_inc_rev = sparse([
            0.0  0.0;   # No gen at bus 1 (load)
            1.0  0.0;   # Gen 1 at bus 2 (cheap)
            0.0  1.0    # Gen 2 at bus 3 (expensive)
        ])
        d_rev = [0.5, 0.05, 0.05]

        # Tight angmin on branch 1 (1→2): power flows 2→1, so A*θ < 0
        angmin_tight = [-0.025, -π, -π]
        net = DCNetwork(n, m, k, A, G_inc_rev, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=fill(π, m), angmin=angmin_tight,
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d_rev)
        sol = solve!(prob)

        # Branch 1 angle difference should be at or near the lower limit
        Aθ = net.A * sol.va
        @test Aθ[1] >= angmin_tight[1] - 1e-4

        # Gamma_lb on branch 1 should be nonzero (binding lower angle limit)
        @test abs(sol.gamma_lb[1]) > 1e-2

        # Gamma duals should be non-negative (standard KKT convention)
        @test all(sol.gamma_lb .>= -1e-6)
        @test all(sol.gamma_ub .>= -1e-6)

        # Full KKT residual should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d_rev)
        idx = kkt_indices(net)
        @test norm(K[idx.gamma_lb]) < 1e-3
        @test norm(K[idx.gamma_ub]) < 1e-3
        @test norm(K) < 1e-3

        # FD verification for demand sensitivity
        dg_dd = calc_sensitivity(prob, :pg, :d)
        dva_dd = calc_sensitivity(prob, :va, :d)
        df_dd = calc_sensitivity(prob, :f, :d)
        dlmp_dd = calc_sensitivity(prob, :lmp, :d)

        @test all(isfinite, Matrix(dg_dd))
        @test all(isfinite, Matrix(dva_dd))
        @test all(isfinite, Matrix(df_dd))
        @test all(isfinite, Matrix(dlmp_dd))

        delta = 1e-5
        for bus in 1:n
            d_pert = copy(d_rev)
            d_pert[bus] += delta

            prob_pert = DCOPFProblem(net, d_pert)
            sol_pert = solve!(prob_pert)

            for (name, S, base, pert) in [
                    ("∂g/∂d", dg_dd, sol.pg, sol_pert.pg),
                    ("∂θ/∂d", dva_dd, sol.va, sol_pert.va),
                    ("∂f/∂d", df_dd, sol.f, sol_pert.f),
                    ("∂lmp/∂d", dlmp_dd, sol.nu_bal, sol_pert.nu_bal)]
                fd = (pert - base) / delta
                analytical = Matrix(S)[:, bus]
                if norm(fd) > 1e-8
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 0.01  # 1% tolerance
                end
            end
        end
    end

    @testset "FD verification with binding angle limits" begin
        angmax_tight = [π, 0.025, π]
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=angmax_tight, angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol_base = solve!(prob)

        # Analytical sensitivities
        dg_dd = calc_sensitivity(prob, :pg, :d)
        dva_dd = calc_sensitivity(prob, :va, :d)
        df_dd = calc_sensitivity(prob, :f, :d)
        dlmp_dd = calc_sensitivity(prob, :lmp, :d)

        @test all(isfinite, Matrix(dg_dd))
        @test all(isfinite, Matrix(dva_dd))
        @test all(isfinite, Matrix(df_dd))
        @test all(isfinite, Matrix(dlmp_dd))

        # Finite-difference: perturb demand at each bus
        delta = 1e-5
        for bus in 1:n
            d_pert = copy(d)
            d_pert[bus] += delta

            prob_pert = DCOPFProblem(net, d_pert)
            sol_pert = solve!(prob_pert)

            for (name, S, base, pert) in [
                    ("∂g/∂d", dg_dd, sol_base.pg, sol_pert.pg),
                    ("∂θ/∂d", dva_dd, sol_base.va, sol_pert.va),
                    ("∂f/∂d", df_dd, sol_base.f, sol_pert.f),
                    ("∂lmp/∂d", dlmp_dd, sol_base.nu_bal, sol_pert.nu_bal)]
                fd = (pert - base) / delta
                analytical = Matrix(S)[:, bus]
                if norm(fd) > 1e-8
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 0.01  # 1% tolerance
                end
            end
        end
    end

    @testset "Participation factors still sum to 1" begin
        angmax_tight = [π, 0.025, π]
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=angmax_tight, angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        solve!(prob)

        dg_dd = calc_sensitivity(prob, :pg, :d)
        dpsh_dd = calc_sensitivity(prob, :psh, :d)

        for j in 1:n
            total = sum(Matrix(dg_dd)[:, j]) + sum(Matrix(dpsh_dd)[:, j])
            @test abs(total - 1.0) < 1e-4
        end
    end

    @testset "case5 with tight angle limits" begin
        net_dict = load_test_case("case5.m")

        dc_net = DCNetwork(net_dict)
        d_case = calc_demand_vector(net_dict)

        # Tighten angle limits on a few branches
        dc_net.angmax[1] = 0.05
        dc_net.angmax[3] = 0.05

        prob = DCOPFProblem(dc_net, d_case)
        sol = solve!(prob)

        # At least one gamma_ub should be nonzero
        @test any(abs.(sol.gamma_ub) .> 1e-4)

        # Gamma duals should be non-negative
        @test all(sol.gamma_ub .>= -1e-6)
        @test all(sol.gamma_lb .>= -1e-6)

        # KKT gamma blocks should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d_case)
        idx = kkt_indices(dc_net)
        @test norm(K[idx.gamma_lb]) < 1e-3
        @test norm(K[idx.gamma_ub]) < 1e-3
        @test norm(K) < 1e-3  # full KKT residual

        # FD verification for demand sensitivity
        dg_dd = calc_sensitivity(prob, :pg, :d)
        @test all(isfinite, Matrix(dg_dd))

        bus_idx = findfirst(d_case .> 0)
        delta = 1e-5
        d_pert = copy(d_case)
        d_pert[bus_idx] += delta

        prob_pert = DCOPFProblem(dc_net, d_pert)
        sol_pert = solve!(prob_pert)

        dg_fd = (sol_pert.pg - sol.pg) / delta
        if norm(dg_fd) > 1e-8
            rel_err = norm(Matrix(dg_dd)[:, bus_idx] - dg_fd) / norm(dg_fd)
            @test rel_err < 0.05
        end
    end
end
