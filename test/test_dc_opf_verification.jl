# DC OPF Finite-Difference Verification Tests
#
# Verifies analytical sensitivities against finite differences for DC OPF
# parameter combinations not covered elsewhere.
#
# Primal FD checks (θ, g, f) use a 2-bus network at 5% tolerance.
# LMP FD checks use case5 where the solver gives precise duals (10% tolerance).

using PowerModelsDiff
using LinearAlgebra
using SparseArrays
using Test

@testset "DC OPF Finite-Difference Verification" begin
    # Base 2-bus network setup (matches existing pattern from runtests.jl)
    n, m, k = 2, 1, 1
    A = sparse([1.0 -1.0])
    G_inc = sparse(reshape([1.0, 0.0], 2, 1))
    b_base = [-10.0]
    d = [0.0, 1.0]

    # Shared parameters
    fmax_base = [100.0]
    gmax = [10.0]
    gmin = [0.0]
    cl_base = [10.0]
    cq_base = [1.0]
    τ = 0.01

    delta = 1e-5
    primal_tol = 0.05   # 5% for primal variables (θ, g, f)

    function make_net(; b=b_base, fmax=fmax_base, cl=cl_base, cq=cq_base)
        DCNetwork(n, m, k, A, G_inc, b;
            fmax=fmax, gmax=gmax, gmin=gmin,
            cl=cl, cq=cq, ref_bus=1, τ=τ)
    end

    function solve_opf(net, demand=d)
        prob = DCOPFProblem(net, demand)
        sol = solve!(prob)
        return prob, sol
    end

    # Helper: compare analytical column vs finite difference for primal variables
    function check_primal(analytical_matrix, sol_base, sol_pert, extract_fn, col_idx)
        fd = (extract_fn(sol_pert) - extract_fn(sol_base)) / delta
        analytical_col = Matrix(analytical_matrix)[:, col_idx]
        if norm(fd) > 1e-10
            rel_err = norm(analytical_col - fd) / norm(fd)
            @test rel_err < primal_tol
        else
            @test norm(analytical_col) < 1e-6
        end
    end

    # =========================================================================
    # :cq (quadratic cost) — primal operands
    # =========================================================================
    @testset "w.r.t. :cq" begin
        net_base = make_net()
        prob_base, sol_base = solve_opf(net_base)

        cq_pert = copy(cq_base); cq_pert[1] += delta
        net_pert = make_net(cq=cq_pert)
        _, sol_pert = solve_opf(net_pert)

        dva_dcq = calc_sensitivity(prob_base, :va, :cq)
        dpg_dcq = calc_sensitivity(prob_base, :pg, :cq)
        df_dcq  = calc_sensitivity(prob_base, :f, :cq)

        check_primal(dva_dcq, sol_base, sol_pert, s -> s.θ, 1)
        check_primal(dpg_dcq, sol_base, sol_pert, s -> s.g, 1)
        check_primal(df_dcq, sol_base, sol_pert, s -> s.f, 1)
    end

    # =========================================================================
    # :fmax (flow limits) — use reduced demand so fmax=0.8 is feasible
    # =========================================================================
    @testset "w.r.t. :fmax" begin
        d_fmax = [0.0, 0.5]  # Demand within flow limit
        net_base = make_net(fmax=[0.8])
        prob_base, sol_base = solve_opf(net_base, d_fmax)

        fmax_pert = [0.8 + delta]
        net_pert = make_net(fmax=fmax_pert)
        _, sol_pert = solve_opf(net_pert, d_fmax)

        dva_dfmax = calc_sensitivity(prob_base, :va, :fmax)
        dpg_dfmax = calc_sensitivity(prob_base, :pg, :fmax)
        df_dfmax  = calc_sensitivity(prob_base, :f, :fmax)

        check_primal(dva_dfmax, sol_base, sol_pert, s -> s.θ, 1)
        check_primal(dpg_dfmax, sol_base, sol_pert, s -> s.g, 1)
        check_primal(df_dfmax, sol_base, sol_pert, s -> s.f, 1)
    end

    # =========================================================================
    # :b (susceptances) — primal operands
    # =========================================================================
    @testset "w.r.t. :b" begin
        net_base = make_net()
        prob_base, sol_base = solve_opf(net_base)

        b_pert = copy(b_base); b_pert[1] += delta
        net_pert = make_net(b=b_pert)
        _, sol_pert = solve_opf(net_pert)

        dva_db = calc_sensitivity(prob_base, :va, :b)
        dpg_db = calc_sensitivity(prob_base, :pg, :b)
        df_db  = calc_sensitivity(prob_base, :f, :b)

        check_primal(dva_db, sol_base, sol_pert, s -> s.θ, 1)
        check_primal(dpg_db, sol_base, sol_pert, s -> s.g, 1)
        check_primal(df_db, sol_base, sol_pert, s -> s.f, 1)
    end

    # =========================================================================
    # :cl (linear cost) — :va and :f operands (not covered by existing tests)
    # =========================================================================
    @testset ":va and :f w.r.t. :cl" begin
        net_base = make_net()
        prob_base, sol_base = solve_opf(net_base)

        dva_dcl = calc_sensitivity(prob_base, :va, :cl)
        df_dcl  = calc_sensitivity(prob_base, :f, :cl)

        cl_pert = copy(cl_base); cl_pert[1] += delta
        net_pert = make_net(cl=cl_pert)
        _, sol_pert = solve_opf(net_pert)

        check_primal(dva_dcl, sol_base, sol_pert, s -> s.θ, 1)
        check_primal(df_dcl, sol_base, sol_pert, s -> s.f, 1)
    end
end

# =============================================================================
# LMP Finite-Difference Verification (case5)
#
# Uses case5 where the solver gives precise duals, unlike the degenerate
# 2-bus topology above. Follows the pattern from runtests.jl demand FD tests.
# =============================================================================
@testset "LMP Finite-Difference (case5)" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping LMP FD tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        delta = 1e-5
        lmp_tol = 0.10  # 10% for LMP (dual variables)

        # Base solve
        prob_base = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob_base)
        lmp_base = calc_lmp(sol_base, dc_net)

        # -----------------------------------------------------------------
        # LMP w.r.t. :d — perturb demand at a load bus
        # -----------------------------------------------------------------
        @testset "LMP w.r.t. :d" begin
            dlmp_dd = calc_sensitivity(prob_base, :lmp, :d)

            bus_idx = findfirst(d .> 0)
            if isnothing(bus_idx)
                bus_idx = 1
            end

            d_pert = copy(d)
            d_pert[bus_idx] += delta
            prob_pert = DCOPFProblem(dc_net, d_pert)
            sol_pert = solve!(prob_pert)
            lmp_pert = calc_lmp(sol_pert, dc_net)

            fd_lmp = (lmp_pert - lmp_base) / delta
            if norm(fd_lmp) > 1e-10
                rel_err = norm(Matrix(dlmp_dd)[:, bus_idx] - fd_lmp) / norm(fd_lmp)
                @test rel_err < lmp_tol
            end
        end

        # -----------------------------------------------------------------
        # LMP w.r.t. :z — perturb a switching state
        # -----------------------------------------------------------------
        @testset "LMP w.r.t. :z" begin
            dlmp_dz = calc_sensitivity(prob_base, :lmp, :z)

            # Perturb switching state of branch 1
            branch_idx = 1
            z_pert = copy(dc_net.z)
            z_pert[branch_idx] -= delta

            # Rebuild DCNetwork with perturbed z
            dc_net_pert = DCNetwork(net)
            dc_net_pert.z .= z_pert

            prob_pert = DCOPFProblem(dc_net_pert, d)
            sol_pert = solve!(prob_pert)
            lmp_pert = calc_lmp(sol_pert, dc_net_pert)

            # Negative perturbation, so fd = (base - pert) / delta
            fd_lmp = (lmp_base - lmp_pert) / delta
            if norm(fd_lmp) > 1e-10
                rel_err = norm(Matrix(dlmp_dz)[:, branch_idx] - fd_lmp) / norm(fd_lmp)
                @test rel_err < lmp_tol
            end
        end

        # -----------------------------------------------------------------
        # LMP w.r.t. :cl — perturb linear cost of a marginal generator
        # -----------------------------------------------------------------
        @testset "LMP w.r.t. :cl" begin
            dlmp_dcl = calc_sensitivity(prob_base, :lmp, :cl)

            # Find a marginal generator (interior, not at bounds)
            gen_idx = findfirst(i ->
                sol_base.g[i] > dc_net.gmin[i] + 0.01 &&
                sol_base.g[i] < dc_net.gmax[i] - 0.01, 1:dc_net.k)
            @test !isnothing(gen_idx)

            net_pert = load_test_case("case5.m")
            net_pert["gen"][string(gen_idx)]["cost"][2] += delta
            dc_net_pert = DCNetwork(net_pert)
            prob_pert = DCOPFProblem(dc_net_pert, d)
            sol_pert = solve!(prob_pert)
            lmp_pert = calc_lmp(sol_pert, dc_net_pert)

            fd_lmp = (lmp_pert - lmp_base) / delta
            if norm(fd_lmp) > 1e-10
                rel_err = norm(Matrix(dlmp_dcl)[:, gen_idx] - fd_lmp) / norm(fd_lmp)
                @test rel_err < lmp_tol
            end
        end

        # -----------------------------------------------------------------
        # LMP w.r.t. :cq — perturb quadratic cost of a marginal generator
        # -----------------------------------------------------------------
        @testset "LMP w.r.t. :cq" begin
            dlmp_dcq = calc_sensitivity(prob_base, :lmp, :cq)

            # Find a marginal generator (interior, not at bounds)
            gen_idx = findfirst(i ->
                sol_base.g[i] > dc_net.gmin[i] + 0.01 &&
                sol_base.g[i] < dc_net.gmax[i] - 0.01, 1:dc_net.k)
            @test !isnothing(gen_idx)

            net_pert = load_test_case("case5.m")
            net_pert["gen"][string(gen_idx)]["cost"][1] += delta
            dc_net_pert = DCNetwork(net_pert)
            prob_pert = DCOPFProblem(dc_net_pert, d)
            sol_pert = solve!(prob_pert)
            lmp_pert = calc_lmp(sol_pert, dc_net_pert)

            fd_lmp = (lmp_pert - lmp_base) / delta
            if norm(fd_lmp) > 1e-10
                rel_err = norm(Matrix(dlmp_dcq)[:, gen_idx] - fd_lmp) / norm(fd_lmp)
                @test rel_err < lmp_tol
            end
        end

        # -----------------------------------------------------------------
        # LMP w.r.t. :fmax — perturb flow limit of a branch
        # -----------------------------------------------------------------
        @testset "LMP w.r.t. :fmax" begin
            dlmp_dfmax = calc_sensitivity(prob_base, :lmp, :fmax)

            # Pick the branch with the most binding flow constraint
            flow_ratio = abs.(sol_base.f) ./ dc_net.fmax
            branch_idx = argmax(flow_ratio)

            net_pert = load_test_case("case5.m")
            dc_net_pert = DCNetwork(net_pert)
            dc_net_pert.fmax[branch_idx] += delta
            prob_pert = DCOPFProblem(dc_net_pert, d)
            sol_pert = solve!(prob_pert)
            lmp_pert = calc_lmp(sol_pert, dc_net_pert)

            fd_lmp = (lmp_pert - lmp_base) / delta
            if norm(fd_lmp) > 1e-10
                rel_err = norm(Matrix(dlmp_dfmax)[:, branch_idx] - fd_lmp) / norm(fd_lmp)
                @test rel_err < lmp_tol
            end
        end

        # -----------------------------------------------------------------
        # LMP w.r.t. :b — perturb susceptance of a branch
        # -----------------------------------------------------------------
        @testset "LMP w.r.t. :b" begin
            dlmp_db = calc_sensitivity(prob_base, :lmp, :b)

            # Perturb susceptance of branch 1
            branch_idx = 1
            net_pert = load_test_case("case5.m")
            # Perturb the reactance to change susceptance: b = -x/(r² + x²)
            # Easier to rebuild DCNetwork with direct b perturbation
            dc_net_pert = DCNetwork(net_pert)
            dc_net_pert.b[branch_idx] += delta
            prob_pert = DCOPFProblem(dc_net_pert, d)
            sol_pert = solve!(prob_pert)
            lmp_pert = calc_lmp(sol_pert, dc_net_pert)

            fd_lmp = (lmp_pert - lmp_base) / delta
            if norm(fd_lmp) > 1e-10
                rel_err = norm(Matrix(dlmp_db)[:, branch_idx] - fd_lmp) / norm(fd_lmp)
                @test rel_err < lmp_tol
            end
        end

    end
end
