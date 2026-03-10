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

using PowerDiff
using PowerModels
using Test

@testset "Unified Architecture" begin
    # Load a test network
    case_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower", "case5.m")
    data = PowerModels.parse_file(case_path)
    net_data = PowerModels.make_basic_network(data)

    @testset "Abstract Type Hierarchy" begin
        # Test that types inherit correctly
        net = DCNetwork(net_data)
        @test net isa AbstractPowerNetwork

        demand = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, demand)
        sol = solve!(prob)
        @test sol isa AbstractOPFSolution
        @test sol isa AbstractPowerFlowState

        pf_state = DCPowerFlowState(net, demand)
        @test pf_state isa AbstractPowerFlowState
    end

    @testset "DC Power Flow State" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)

        # Test construction
        pf_state = DCPowerFlowState(net, demand)
        @test length(pf_state.va) == net.n
        @test length(pf_state.f) == net.m
        @test pf_state.p == -demand  # Since g = 0

        # Test with generation
        g = zeros(net.n)
        g[1] = sum(demand)  # All generation at bus 1
        pf_state2 = DCPowerFlowState(net, g, demand)
        @test pf_state2.pg == g
        @test pf_state2.d == demand
        @test pf_state2.p == g - demand
    end

    @testset "DC Power Flow Switching Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, demand)

        dva_dsw = calc_sensitivity(pf_state, :va, :sw)
        @test dva_dsw isa Sensitivity
        @test dva_dsw.formulation == :dcpf
        @test dva_dsw.operand == :va
        @test dva_dsw.parameter == :sw
        @test size(dva_dsw) == (net.n, net.m)

        df_dsw = calc_sensitivity(pf_state, :f, :sw)
        @test df_dsw isa Sensitivity
        @test df_dsw.formulation == :dcpf
        @test df_dsw.operand == :f
        @test df_dsw.parameter == :sw
        @test size(df_dsw) == (net.m, net.m)
    end

    @testset "DC Power Flow Demand Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, demand)

        dva_dd = calc_sensitivity(pf_state, :va, :d)
        @test dva_dd isa Sensitivity
        @test dva_dd.formulation == :dcpf
        @test dva_dd.operand == :va
        @test dva_dd.parameter == :d
        @test size(dva_dd) == (net.n, net.n)

        df_dd = calc_sensitivity(pf_state, :f, :d)
        @test df_dd isa Sensitivity
        @test df_dd.formulation == :dcpf
        @test df_dd.operand == :f
        @test df_dd.parameter == :d
        @test size(df_dd) == (net.m, net.n)
    end

    @testset "DC OPF Switching Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, demand)

        dva_dsw = calc_sensitivity(prob, :va, :sw)
        @test dva_dsw isa Sensitivity
        @test dva_dsw.formulation == :dcopf
        @test size(dva_dsw) == (net.n, net.m)

        dg_dsw = calc_sensitivity(prob, :pg, :sw)
        @test dg_dsw isa Sensitivity
        @test dg_dsw.formulation == :dcopf
        @test dg_dsw.operand == :pg
        @test size(dg_dsw) == (net.k, net.m)

        df_dsw = calc_sensitivity(prob, :f, :sw)
        @test df_dsw isa Sensitivity
        @test df_dsw.formulation == :dcopf
        @test size(df_dsw) == (net.m, net.m)
    end

    @testset "DC OPF Demand Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, demand)

        dlmp_dd = calc_sensitivity(prob, :lmp, :d)
        @test dlmp_dd isa Sensitivity
        @test dlmp_dd.formulation == :dcopf
        @test dlmp_dd.operand == :lmp
        @test dlmp_dd.parameter == :d
        @test size(dlmp_dd) == (net.n, net.n)

        # Test index mappings
        @test length(dlmp_dd.row_to_id) == net.n
        @test length(dlmp_dd.col_to_id) == net.n

        # Test matrix interface
        @test dlmp_dd[1, 1] isa Float64
        @test Matrix(dlmp_dd) isa Matrix{Float64}
    end

    @testset "ACNetwork" begin
        # Construct from PowerModels network
        ac_net = ACNetwork(net_data)
        @test ac_net isa AbstractPowerNetwork
        @test ac_net.n == length(net_data["bus"])
        @test ac_net.m == length(net_data["branch"])

        # Admittance matrix reconstruction
        Y = admittance_matrix(ac_net)
        @test size(Y) == (ac_net.n, ac_net.n)

        # With switching
        sw = ones(ac_net.m)
        sw[1] = 0.0  # Open first branch
        Y_switched = admittance_matrix(ac_net, sw)
        @test Y_switched != Y
    end

    @testset "ACPowerFlowState" begin
        # Solve AC power flow
        PowerModels.compute_ac_pf!(net_data)

        # Construct state from network
        state = ACPowerFlowState(net_data)
        @test state isa AbstractPowerFlowState
        @test length(state.v) == state.n
        @test size(state.Y) == (state.n, state.n)

        # Check generation/demand are extracted
        @test length(state.pg) == state.n
        @test length(state.pd) == state.n
        @test length(state.qg) == state.n
        @test length(state.qd) == state.n
    end

    @testset "AC Power Flow Sensitivity" begin
        # Solve AC power flow
        PowerModels.compute_ac_pf!(net_data)
        state = ACPowerFlowState(net_data)

        dvm_dp = calc_sensitivity(state, :vm, :p)
        @test dvm_dp isa Sensitivity
        @test dvm_dp.formulation == :acpf
        @test dvm_dp.operand == :vm
        @test dvm_dp.parameter == :p
        @test size(dvm_dp) == (state.n, state.n)

        dvm_dq = calc_sensitivity(state, :vm, :q)
        @test dvm_dq isa Sensitivity
        @test dvm_dq.formulation == :acpf
        @test dvm_dq.operand == :vm
        @test dvm_dq.parameter == :q
        @test size(dvm_dq) == (state.n, state.n)

        # Sensitivities should be real and finite
        @test all(isfinite, Matrix(dvm_dp))
        @test all(isfinite, Matrix(dvm_dq))
    end

    @testset "Symbol Introspection" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)

        pf_state = DCPowerFlowState(net, demand)
        @test operand_symbols(pf_state) == [:va, :f]
        @test parameter_symbols(pf_state) == [:d, :sw]

        prob = DCOPFProblem(net, demand)
        @test operand_symbols(prob) == [:va, :pg, :f, :psh, :lmp]
        @test parameter_symbols(prob) == [:d, :sw, :cq, :cl, :fmax, :b]

        PowerModels.compute_ac_pf!(net_data)
        ac_state = ACPowerFlowState(net_data)
        @test Set(operand_symbols(ac_state)) == Set([:vm, :v, :im, :va, :f, :p, :q])
        @test Set(parameter_symbols(ac_state)) == Set([:p, :q, :va, :vm, :d, :qd])

        ac_prob = ACOPFProblem(net_data)
        @test Set(operand_symbols(ac_prob)) == Set([:vm, :va, :pg, :qg, :lmp])
        @test Set(parameter_symbols(ac_prob)) == Set([:sw, :d, :qd, :cq, :cl, :fmax])
    end

    @testset "Base.show Methods" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, demand)
        prob = DCOPFProblem(net, demand)
        sol = solve!(prob)

        # One-line show
        @test sprint(show, net) isa String
        @test sprint(show, pf_state) isa String
        @test sprint(show, prob) isa String
        @test sprint(show, sol) isa String

        # Multi-line show (MIME"text/plain")
        @test sprint(show, MIME("text/plain"), net) isa String
        @test sprint(show, MIME("text/plain"), pf_state) isa String
        @test sprint(show, MIME("text/plain"), prob) isa String
        @test sprint(show, MIME("text/plain"), sol) isa String
    end

    @testset "Property Aliases" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)

        # DCNetwork aliases
        @test net.τ === net.tau
        @test net.Δθ_max === net.angmax
        @test net.Δθ_min === net.angmin

        # DCPowerFlowState aliases
        pf_state = DCPowerFlowState(net, demand)
        @test pf_state.θ === pf_state.va
        @test pf_state.g === pf_state.pg

        # DCOPFSolution aliases
        prob = DCOPFProblem(net, demand)
        sol = solve!(prob)
        @test sol.θ === sol.va
        @test sol.g === sol.pg
        @test sol.ν_bal === sol.nu_bal
        @test sol.λ_ub === sol.lam_ub
        @test sol.ρ_ub === sol.rho_ub
        @test sol.μ_lb === sol.mu_lb

        # DCOPFProblem aliases
        @test prob.θ === prob.va
        @test prob.g === prob.pg
    end

    @testset "Sensitivity Metadata" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, demand)

        # Get sensitivity and check metadata
        sens = calc_sensitivity(prob, :lmp, :d)
        @test sens.formulation == :dcopf
        @test sens.operand == :lmp
        @test sens.parameter == :d

        # Test that metadata is accessible as fields
        dva_dsw = calc_sensitivity(prob, :va, :sw)
        @test dva_dsw.formulation == :dcopf
        @test dva_dsw.operand == :va
        @test dva_dsw.parameter == :sw
    end
end

println("All unified architecture tests passed!")
