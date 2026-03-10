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

# Parameter Transform Tests
#
# Verifies that type-specific parameter transforms work correctly:
# - ACPowerFlowState: ∂/∂d = -∂/∂p, ∂/∂qd = -∂/∂q
# - DCPowerFlowState/DCOPFProblem: :d is native (no transform needed)
# - Introspection includes transform-derived symbols

using PowerDiff
using PowerModels
using LinearAlgebra
using Test

@testset "Parameter Transforms" begin
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")
    pm_data = PowerModels.parse_file(file)
    net_data = PowerModels.make_basic_network(pm_data)

    @testset "AC PF: ∂/∂d = -∂/∂p (algebraic)" begin
        pf_data = deepcopy(net_data)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        # Test all operands that work with :p
        for op in [:vm, :v, :im, :va, :f]
            S_p = calc_sensitivity(state, op, :p)
            S_d = calc_sensitivity(state, op, :d)
            @test Matrix(S_d) ≈ -Matrix(S_p) atol=1e-12
            @test S_d.parameter == :d
            @test S_d.formulation == :acpf
        end
    end

    @testset "AC PF: ∂/∂qd = -∂/∂q (algebraic)" begin
        pf_data = deepcopy(net_data)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        for op in [:vm, :v, :im, :va, :f]
            S_q = calc_sensitivity(state, op, :q)
            S_qd = calc_sensitivity(state, op, :qd)
            @test Matrix(S_qd) ≈ -Matrix(S_q) atol=1e-12
            @test S_qd.parameter == :qd
        end
    end

    @testset "AC PF: Jacobian blocks + demand transform is invalid" begin
        pf_data = deepcopy(net_data)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        # :p and :q as operands don't have :p/:q as native params,
        # so the :d/:qd transforms cannot apply
        @test_throws ArgumentError calc_sensitivity(state, :p, :d)
        @test_throws ArgumentError calc_sensitivity(state, :q, :d)
        @test_throws ArgumentError calc_sensitivity(state, :p, :qd)
        @test_throws ArgumentError calc_sensitivity(state, :q, :qd)
    end

    @testset "DC PF: :d is native, no transform interference" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        pf = DCPowerFlowState(net, d)

        # :d should work directly (native)
        S = calc_sensitivity(pf, :va, :d)
        @test S isa Sensitivity
        @test S.formulation == :dcpf
        @test S.parameter == :d
    end

    @testset "DC OPF: :d is native, no transform interference" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        solve!(prob)

        S = calc_sensitivity(prob, :lmp, :d)
        @test S isa Sensitivity
        @test S.formulation == :dcopf
        @test S.parameter == :d
    end

    @testset "Introspection includes transform-derived symbols" begin
        pf_data = deepcopy(net_data)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        params = parameter_symbols(state)
        @test :d in params
        @test :qd in params
        @test :p in params
        @test :q in params

        ops = operand_symbols(state)
        @test :vm in ops
        @test :va in ops
        @test :f in ops
        @test :p in ops
        @test :q in ops
    end

    @testset "DC PF: introspection excludes :qd" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        pf = DCPowerFlowState(net, d)

        params = parameter_symbols(pf)
        @test :d in params
        @test :qd ∉ params
    end
end
