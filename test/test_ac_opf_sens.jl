# Test AC OPF switching sensitivity

using PowerModelsDiff
using PowerModels
using LinearAlgebra
using Test

@testset "AC OPF Switching Sensitivity" begin
    # Load test case
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")

    pm_data = PowerModels.parse_file(file)
    pm_data = PowerModels.make_basic_network(pm_data)

    # Create and solve AC OPF
    @testset "ACOPFProblem construction and solving" begin
        prob = ACOPFProblem(pm_data; silent=true)

        @test prob.network.n == 5
        @test prob.network.m == 7
        @test prob.n_gen == 5

        sol = solve!(prob)

        @test sol.objective > 0
        @test length(sol.vm) == 5
        @test length(sol.va) == 5
        @test length(sol.pg) == 5
        @test length(sol.qg) == 5

        # Voltage magnitudes should be within limits
        @test all(sol.vm .>= 0.9)
        @test all(sol.vm .<= 1.1)

    end

    @testset "Switching sensitivity computation" begin
        prob = ACOPFProblem(pm_data; silent=true)

        dvm_dsw = calc_sensitivity(prob, :vm, :sw)
        dva_dsw = calc_sensitivity(prob, :va, :sw)
        dpg_dsw = calc_sensitivity(prob, :pg, :sw)
        dqg_dsw = calc_sensitivity(prob, :qg, :sw)

        @test size(dvm_dsw) == (5, 7)
        @test size(dva_dsw) == (5, 7)
        @test size(dpg_dsw) == (5, 7)
        @test size(dqg_dsw) == (5, 7)

        # Sensitivities should be finite
        @test all(isfinite.(Matrix(dvm_dsw)))
        @test all(isfinite.(Matrix(dva_dsw)))
        @test all(isfinite.(Matrix(dpg_dsw)))
        @test all(isfinite.(Matrix(dqg_dsw)))

    end

    @testset "Symbol-based API" begin
        prob = ACOPFProblem(pm_data; silent=true)

        dvm_dsw = calc_sensitivity(prob, :vm, :sw)
        @test size(dvm_dsw) == (5, 7)

        dva_dsw = calc_sensitivity(prob, :va, :sw)
        @test size(dva_dsw) == (5, 7)

        dpg_dsw = calc_sensitivity(prob, :pg, :sw)
        @test size(dpg_dsw) == (5, 7)

        dqg_dsw = calc_sensitivity(prob, :qg, :sw)
        @test size(dqg_dsw) == (5, 7)

    end

    @testset "KKT residual at optimum" begin
        prob = ACOPFProblem(pm_data; silent=true)
        sol = solve!(prob)
        z0 = ac_flatten_variables(sol, prob)
        K = ac_kkt(z0, prob)
        @test length(K) == ac_kkt_dims(prob)

        # Full KKT residual should be small (bounded by solver tolerance)
        @test norm(K) < 1e-2

        # Individual components
        idx = ac_kkt_indices(prob)
        @test norm(K[idx.va]) < 1e-2          # va stationarity
        @test norm(K[idx.vm]) < 1e-2          # vm stationarity
        @test norm(K[idx.pg]) < 1e-6          # pg stationarity (exact: linear)
        @test norm(K[idx.qg]) < 1e-6          # qg stationarity (exact: linear)
        @test norm(K[idx.nu_p_bal]) < 1e-6     # power balance
        @test norm(K[idx.nu_q_bal]) < 1e-6     # reactive balance
        @test norm(K[idx.nu_ref_bus]) < 1e-6   # reference bus
    end

    @testset "Finite-difference verification" begin
        prob = ACOPFProblem(pm_data; silent=true)
        dvm_dsw = calc_sensitivity(prob, :vm, :sw)
        dpg_dsw = calc_sensitivity(prob, :pg, :sw)
        dva_dsw = calc_sensitivity(prob, :va, :sw)
        dqg_dsw = calc_sensitivity(prob, :qg, :sw)
        sol_base = prob.cache.solution

        ε = 1e-5
        for e in 1:min(3, prob.network.m)
            # Build perturbed problem with sw[e] -= ε baked into JuMP model
            net_pert = ACNetwork(pm_data)
            net_pert.sw[e] -= ε
            prob_pert = ACOPFProblem(net_pert, pm_data; silent=true)
            sol_pert = solve!(prob_pert)

            fd_dvm = (sol_base.vm - sol_pert.vm) / ε
            fd_dpg = (sol_base.pg - sol_pert.pg) / ε
            fd_dva = (sol_base.va - sol_pert.va) / ε
            fd_dqg = (sol_base.qg - sol_pert.qg) / ε

            # Verify voltage magnitude sensitivities
            if norm(fd_dvm) > 1e-10
                rel_error = norm(Matrix(dvm_dsw)[:, e] - fd_dvm) / norm(fd_dvm)
                @test rel_error < 1e-3
            end

            # Verify generation sensitivities
            if norm(fd_dpg) > 1e-10
                rel_error = norm(Matrix(dpg_dsw)[:, e] - fd_dpg) / norm(fd_dpg)
                @test rel_error < 1e-3
            end

            # Verify voltage angle sensitivities
            if norm(fd_dva) > 1e-10
                rel_error = norm(Matrix(dva_dsw)[:, e] - fd_dva) / norm(fd_dva)
                @test rel_error < 1e-3
            end

            # Verify reactive generation sensitivities
            if norm(fd_dqg) > 1e-10
                rel_error = norm(Matrix(dqg_dsw)[:, e] - fd_dqg) / norm(fd_dqg)
                @test rel_error < 1e-3
            end
        end
    end

    @testset "Cache reuse across operands" begin
        prob = ACOPFProblem(pm_data; silent=true)

        # First call computes and caches
        dvm_dsw = calc_sensitivity(prob, :vm, :sw)
        @test size(dvm_dsw) == (5, 7)
        @test !isnothing(prob.cache.dz_dsw)

        # Subsequent calls reuse cache (no re-solve)
        dva_dsw = calc_sensitivity(prob, :va, :sw)
        dpg_dsw = calc_sensitivity(prob, :pg, :sw)
        dqg_dsw = calc_sensitivity(prob, :qg, :sw)
        @test size(dva_dsw) == (5, 7)
        @test size(dpg_dsw) == (5, 7)
        @test size(dqg_dsw) == (5, 7)
    end
end
