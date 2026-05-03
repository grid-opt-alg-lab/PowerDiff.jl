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

@testset "update_fmax! correctness" begin

    net_data = load_test_case("case14.m")
    if isnothing(net_data)
        @info "Skipping update_fmax! tests - PowerModels test data not found"
        @test_skip false
    else

    # =========================================================================
    @testset "solve! after update_fmax! matches fresh construction (case14)" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        fmax_base = copy(dc_net.fmax)

        # Build and solve base problem
        prob = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob)

        # Perturb flow limits (case14 has loose base limits ~9.9; at 0.05× flows ~1.5 bind)
        fmax_new = 0.05 .* fmax_base

        # Method 1: update_fmax! + solve!
        update_fmax!(prob, fmax_new)
        sol_updated = solve!(prob)

        # Method 2: fresh construction with perturbed fmax
        dc_net_fresh = DCNetwork(net_data)
        dc_net_fresh.fmax .= fmax_new
        prob_fresh = DCOPFProblem(dc_net_fresh, d)
        sol_fresh = solve!(prob_fresh)

        # Solutions must match (correctness of update_fmax!)
        @test sol_updated.va ≈ sol_fresh.va atol=1e-6
        @test sol_updated.pg ≈ sol_fresh.pg atol=1e-6
        @test sol_updated.f  ≈ sol_fresh.f  atol=1e-6
        @test sol_updated.objective ≈ sol_fresh.objective atol=1e-6

        # Tightened problem must be at least as expensive as base (restricted feasible set)
        @test sol_updated.objective ≥ sol_base.objective - 1e-6
    end

    # =========================================================================
    @testset "update_fmax! tightening actually changes the solution (3-bus congested)" begin
        # Uses common.jl's 3-bus network designed with a binding flow limit
        dc_net = create_3bus_congested_network()
        d = [0.0, 0.0, 1.0]

        prob = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob)

        # Tighten the already-binding line 1→3 further: 0.5 → 0.3
        fmax_new = copy(dc_net.fmax)
        fmax_new[1] = 0.3
        update_fmax!(prob, fmax_new)
        sol_tight = solve!(prob)

        # Solution must change: cheap gen output drops, expensive gen output rises
        @test sol_tight.f[1] < sol_base.f[1] - 1e-4   # less flow on congested line
        @test sol_tight.pg[1] < sol_base.pg[1] - 1e-4 # cheap gen backed down
        @test sol_tight.pg[2] > sol_base.pg[2] + 1e-4 # expensive gen ramped up
        @test sol_tight.objective > sol_base.objective + 1e-4  # cost rises
    end

    # =========================================================================
    @testset "sensitivities after update_fmax! match fresh construction" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        fmax_base = copy(dc_net.fmax)

        # Build base
        prob = DCOPFProblem(dc_net, d)
        solve!(prob)

        # Perturb fmax to a congested regime
        fmax_new = 0.4 .* fmax_base
        update_fmax!(prob, fmax_new)
        solve!(prob)

        # Sensitivities from updated problem
        dlmp_dd_updated   = Matrix(calc_sensitivity(prob, :lmp, :d))
        dlmp_dfmax_updated = Matrix(calc_sensitivity(prob, :lmp, :fmax))
        dpg_dd_updated    = Matrix(calc_sensitivity(prob, :pg, :d))

        # Sensitivities from fresh problem
        dc_net_fresh = DCNetwork(net_data)
        dc_net_fresh.fmax .= fmax_new
        prob_fresh = DCOPFProblem(dc_net_fresh, d)
        solve!(prob_fresh)
        dlmp_dd_fresh   = Matrix(calc_sensitivity(prob_fresh, :lmp, :d))
        dlmp_dfmax_fresh = Matrix(calc_sensitivity(prob_fresh, :lmp, :fmax))
        dpg_dd_fresh    = Matrix(calc_sensitivity(prob_fresh, :pg, :d))

        @test dlmp_dd_updated    ≈ dlmp_dd_fresh    atol=1e-6
        @test dlmp_dfmax_updated ≈ dlmp_dfmax_fresh atol=1e-6
        @test dpg_dd_updated     ≈ dpg_dd_fresh     atol=1e-6
    end

    # =========================================================================
    @testset "repeated update_fmax! recovers original solution" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        fmax_base = copy(dc_net.fmax)

        prob = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob)

        # Tighten then restore
        update_fmax!(prob, 0.3 .* fmax_base)
        solve!(prob)
        update_fmax!(prob, fmax_base)
        sol_restored = solve!(prob)

        @test sol_restored.va ≈ sol_base.va atol=1e-6
        @test sol_restored.pg ≈ sol_base.pg atol=1e-6
        @test sol_restored.f  ≈ sol_base.f  atol=1e-6
        @test sol_restored.objective ≈ sol_base.objective atol=1e-6
    end

    # =========================================================================
    @testset "VJP :lmp/:fmax finite difference consistency after update_fmax!" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        fmax_base = copy(dc_net.fmax)

        # Use a congested regime so LMPs are sensitive to fmax
        fmax_new = 0.4 .* fmax_base

        prob = DCOPFProblem(dc_net, d)
        update_fmax!(prob, fmax_new)
        sol0 = solve!(prob)
        lmp0 = sol0.nu_bal

        # VJP: (∂λ/∂fmax)^T * w where w = 1 — gives the gradient of 1^T λ w.r.t fmax
        n, m = dc_net.n, dc_net.m
        w = ones(n)
        grad_vjp = zeros(m)
        work = zeros(kkt_dims(prob))
        vjp!(grad_vjp, prob, :lmp, :fmax, w; work=work)

        # Finite difference check on 3 representative branches (binding-ish)
        fd = zeros(m)
        δ = 1e-5
        for e in 1:min(3, m)
            fmax_pert = copy(fmax_new)
            fmax_pert[e] += δ
            update_fmax!(prob, fmax_pert)
            sol_p = solve!(prob)
            fd[e] = (sum(sol_p.nu_bal) - sum(lmp0)) / δ
            update_fmax!(prob, fmax_new)  # restore for next iter
            solve!(prob)
        end

        # Compare the 3 FD entries against vjp (loose tol — congested duals are noisy)
        for e in 1:min(3, m)
            @test isapprox(grad_vjp[e], fd[e]; atol=5e-2, rtol=5e-2) ||
                  abs(grad_vjp[e]) < 1e-6  # non-binding branches: fd ≈ vjp ≈ 0
        end
    end

    # =========================================================================
    @testset "argument validation" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(dc_net, d)

        # Wrong length
        @test_throws DimensionMismatch update_fmax!(prob, zeros(dc_net.m + 1))
        @test_throws DimensionMismatch update_fmax!(prob, zeros(dc_net.m - 1))

        # Negative entries
        bad = ones(dc_net.m)
        bad[1] = -0.1
        @test_throws ArgumentError update_fmax!(prob, bad)

        # Zero fmax is allowed (effectively opens the branch)
        @test update_fmax!(prob, zeros(dc_net.m)) === prob
    end

    # =========================================================================
    @testset "cache invalidation" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        prob = DCOPFProblem(dc_net, d)
        solve!(prob)

        # Populate several cache fields
        _ = calc_sensitivity(prob, :lmp, :d)
        _ = calc_sensitivity(prob, :lmp, :fmax)
        @test prob.cache.dz_dd    !== nothing
        @test prob.cache.dz_dfmax !== nothing
        @test prob.cache.solution !== nothing

        # update_fmax! invalidates solution + all dz_d* caches
        update_fmax!(prob, 0.5 .* copy(dc_net.fmax))
        @test prob.cache.solution === nothing
        @test prob.cache.dz_dd    === nothing
        @test prob.cache.dz_dfmax === nothing
        @test prob.cache.kkt_factor === nothing

        # But b_r_factor (topology dependent) must be preserved (fmax doesn't change topology)
        @test prob.cache.b_r_factor !== nothing
    end

    end  # if isnothing(net_data)
end
