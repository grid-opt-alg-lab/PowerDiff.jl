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

# =============================================================================
# Tests for update_switching! correctness
# =============================================================================
#
# Verifies update_switching! correctly modifies switching states and produces
# solutions consistent with fresh construction for both DC and AC OPF.
# Also checks that sensitivities computed after update_switching! match
# those from a freshly-constructed problem with the same switching state.

@testset "update_switching! correctness" begin

    # Load test data
    case_path = joinpath(PM_DATA_DIR, "case5.m")
    net_data = load_test_case("case5.m")
    if isnothing(net_data)
        @info "Skipping update_switching! tests - PowerModels test data not found"
        @test_skip false
    else

    # =========================================================================
    @testset "DC OPF: solve! after update_switching! matches fresh construction" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Solve base problem
        prob = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob)

        # Perturb switching state
        sw_new = copy(dc_net.sw)
        sw_new[1] = 0.8

        # Method 1: update_switching! + solve!
        update_switching!(prob, sw_new)
        sol_updated = solve!(prob)

        # Method 2: fresh construction with perturbed sw
        dc_net_fresh = DCNetwork(net_data)
        dc_net_fresh.sw .= sw_new
        prob_fresh = DCOPFProblem(dc_net_fresh, d)
        sol_fresh = solve!(prob_fresh)

        # Solutions must match
        @test sol_updated.va ≈ sol_fresh.va atol=1e-6
        @test sol_updated.pg ≈ sol_fresh.pg atol=1e-6
        @test sol_updated.f ≈ sol_fresh.f atol=1e-6
        @test sol_updated.objective ≈ sol_fresh.objective atol=1e-6

        # Must differ from unperturbed base (proving sw change had effect)
        @test !isapprox(sol_updated.f, sol_base.f, atol=1e-4)
    end

    # =========================================================================
    @testset "DC OPF: sensitivities after update_switching! match fresh" begin
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Build and solve
        prob = DCOPFProblem(dc_net, d)
        solve!(prob)

        # Perturb switching state
        sw_new = copy(dc_net.sw)
        sw_new[1] = 0.8
        update_switching!(prob, sw_new)

        # Sensitivities from updated problem
        dva_dsw_updated = Matrix(calc_sensitivity(prob, :va, :sw))
        dpg_dd_updated = Matrix(calc_sensitivity(prob, :pg, :d))

        # Sensitivities from fresh problem
        dc_net_fresh = DCNetwork(net_data)
        dc_net_fresh.sw .= sw_new
        prob_fresh = DCOPFProblem(dc_net_fresh, d)
        solve!(prob_fresh)
        dva_dsw_fresh = Matrix(calc_sensitivity(prob_fresh, :va, :sw))
        dpg_dd_fresh = Matrix(calc_sensitivity(prob_fresh, :pg, :d))

        @test dva_dsw_updated ≈ dva_dsw_fresh atol=1e-6
        @test dpg_dd_updated ≈ dpg_dd_fresh atol=1e-6
    end

    # =========================================================================
    @testset "AC OPF: solve! after update_switching! matches fresh construction" begin
        # Build and solve base AC OPF
        prob = ACOPFProblem(deepcopy(net_data))
        sol_base = solve!(prob)

        # Perturb switching state
        sw_new = copy(prob.network.sw)
        sw_new[1] = 0.8

        # Method 1: update_switching! + solve!
        update_switching!(prob, sw_new)
        sol_updated = solve!(prob)

        # Method 2: fresh construction with perturbed sw, then update_switching!
        prob_fresh = ACOPFProblem(deepcopy(net_data))
        update_switching!(prob_fresh, sw_new)
        sol_fresh = solve!(prob_fresh)

        # Solutions must match
        @test sol_updated.va ≈ sol_fresh.va atol=1e-5
        @test sol_updated.vm ≈ sol_fresh.vm atol=1e-5
        @test sol_updated.pg ≈ sol_fresh.pg atol=1e-5
        @test sol_updated.objective ≈ sol_fresh.objective atol=1e-5

        # Must differ from unperturbed base (proving sw change had effect)
        @test !isapprox(sol_updated.va, sol_base.va, atol=1e-4)
    end

    # =========================================================================
    @testset "AC OPF: sensitivities after update_switching! match fresh" begin
        # Build and solve base AC OPF
        prob = ACOPFProblem(deepcopy(net_data))
        solve!(prob)

        # Perturb switching state
        sw_new = copy(prob.network.sw)
        sw_new[1] = 0.8
        update_switching!(prob, sw_new)

        # Sensitivities from updated problem
        dvm_dsw_updated = Matrix(calc_sensitivity(prob, :vm, :sw))

        # Sensitivities from fresh problem with update_switching!
        prob_fresh = ACOPFProblem(deepcopy(net_data))
        update_switching!(prob_fresh, sw_new)
        solve!(prob_fresh)
        dvm_dsw_fresh = Matrix(calc_sensitivity(prob_fresh, :vm, :sw))

        @test dvm_dsw_updated ≈ dvm_dsw_fresh atol=1e-5
    end

    end  # if isnothing(net_data)
end
