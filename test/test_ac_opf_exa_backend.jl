using PowerDiff
using Ipopt
using JuMP: optimizer_with_attributes
using LinearAlgebra
using Test

import PowerDiff: kkt, flatten_variables

@testset "AC OPF Exa Backend" begin
    pm_data = load_test_case("case5.m")

    if isnothing(pm_data)
        @test_skip false
    else
        function _assert_small_kkt(prob, sol; tol=2e-3)
            K = kkt(flatten_variables(sol, prob), prob)
            @test norm(K, Inf) < tol
        end

        function _assert_solution_parity(prob_exa, sol_exa, prob_jump, sol_jump;
            primal_atol=1e-5, dual_atol=1e-3, price_atol=1e-5, obj_atol=1e-4, rtol=1e-5)
            for field in (:va, :vm, :pg, :qg)
                @test getfield(sol_exa, field) ≈ getfield(sol_jump, field) atol=primal_atol rtol=rtol
            end

            for field in (
                :nu_p_bal, :nu_q_bal, :nu_ref_bus,
                :nu_p_fr, :nu_p_to, :nu_q_fr, :nu_q_to,
                :lam_angle_lb, :lam_angle_ub,
                :mu_vm_lb, :mu_vm_ub,
                :rho_pg_lb, :rho_pg_ub, :rho_qg_lb, :rho_qg_ub,
            )
                @test getfield(sol_exa, field) ≈ getfield(sol_jump, field) atol=dual_atol rtol=rtol
            end

            @test sol_exa.objective ≈ sol_jump.objective atol=obj_atol rtol=1e-8

            for arc in keys(sol_jump.p)
                @test sol_exa.p[arc] ≈ sol_jump.p[arc] atol=primal_atol rtol=rtol
                @test sol_exa.q[arc] ≈ sol_jump.q[arc] atol=primal_atol rtol=rtol
            end

            @test calc_lmp(sol_exa, prob_exa) ≈ calc_lmp(sol_jump, prob_jump) atol=price_atol rtol=rtol
            @test calc_qlmp(sol_exa, prob_exa) ≈ calc_qlmp(sol_jump, prob_jump) atol=price_atol rtol=rtol
        end

        @testset "Constructs and solves" begin
            prob = ACOPFProblem(pm_data; backend=:exa, silent=true)
            sol = solve!(prob)

            @test sol.objective > 0
            @test all(isfinite, sol.va)
            @test all(isfinite, sol.vm)
            @test all(isfinite, sol.pg)
            @test all(isfinite, sol.qg)
            _assert_small_kkt(prob, sol)
        end

        @testset "Rejects custom optimizer" begin
            optimizer = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
            @test_throws ArgumentError ACOPFProblem(pm_data; backend=:exa, optimizer)
        end

        @testset "Solve parity with JuMP backend" begin
            prob_jump = ACOPFProblem(pm_data; backend=:jump, silent=true)
            prob_exa = ACOPFProblem(pm_data; backend=:exa, silent=true)

            sol_jump = solve!(prob_jump)
            sol_exa = solve!(prob_exa)

            _assert_solution_parity(prob_exa, sol_exa, prob_jump, sol_jump)
            _assert_small_kkt(prob_exa, sol_exa)
        end

        @testset "Switching update parity" begin
            prob_jump = ACOPFProblem(pm_data; backend=:jump, silent=true)
            prob_exa = ACOPFProblem(pm_data; backend=:exa, silent=true)

            sw = ones(prob_jump.network.m)
            sw[1] = 0.85
            sw[min(2, end)] = 0.7

            update_switching!(prob_jump, sw)
            update_switching!(prob_exa, sw)

            sol_jump = solve!(prob_jump)
            sol_exa = solve!(prob_exa)

            _assert_solution_parity(prob_exa, sol_exa, prob_jump, sol_jump;
                primal_atol=2e-5, dual_atol=1e-3, obj_atol=1e-4, rtol=2e-5)
            _assert_small_kkt(prob_exa, sol_exa; tol=3e-3)
        end

        @testset "Sensitivity parity" begin
            prob_jump = ACOPFProblem(pm_data; backend=:jump, silent=true)
            prob_exa = ACOPFProblem(pm_data; backend=:exa, silent=true)

            for (op, param, atol, rtol) in [
                (:pg, :d, 1e-4, 1e-4),
                (:va, :sw, 1e-4, 1e-4),
                (:lmp, :d, 5e-4, 5e-4),
            ]
                S_jump = calc_sensitivity(prob_jump, op, param)
                S_exa = calc_sensitivity(prob_exa, op, param)

                @test size(S_exa) == size(S_jump)
                @test S_exa.row_to_id == S_jump.row_to_id
                @test S_exa.col_to_id == S_jump.col_to_id
                @test Matrix(S_exa) ≈ Matrix(S_jump) atol=atol rtol=rtol
            end
        end
    end
end
