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

# Exhaustive sensitivity coverage test
# Verifies that ALL valid (operand, parameter) combinations work for each formulation.

using PowerDiff
using PowerModels
using LinearAlgebra
using Test

@testset "Sensitivity Coverage" begin
    # Load test case
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")
    pm_data = PowerModels.parse_file(file)
    net_data = PowerModels.make_basic_network(pm_data)

    @testset "DC Power Flow — all 6 combinations" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        pf = DCPowerFlowState(net, d)

        combos = [
            (:va, :d, (net.n, net.n)),
            (:f,  :d, (net.m, net.n)),
            (:va, :sw, (net.n, net.m)),
            (:f,  :sw, (net.m, net.m)),
            (:va, :b, (net.n, net.m)),
            (:f,  :b, (net.m, net.m)),
        ]

        for (op, param, expected_size) in combos
            @testset "$op w.r.t. $param" begin
                S = calc_sensitivity(pf, op, param)
                @test S isa Sensitivity
                @test S.formulation == :dcpf
                @test size(S) == expected_size
                @test all(isfinite, Matrix(S))
            end
        end

        # Invalid combinations should throw
        @test_throws ArgumentError calc_sensitivity(pf, :lmp, :d)
        @test_throws ArgumentError calc_sensitivity(pf, :pg, :d)
        @test_throws ArgumentError calc_sensitivity(pf, :vm, :d)
    end

    @testset "DC OPF — all 30 combinations" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        solve!(prob)

        operands = [:va, :pg, :f, :psh, :lmp]
        params = [:d, :sw, :cl, :cq, :fmax, :b]

        # Expected sizes for each operand
        op_sizes = Dict(:va => net.n, :pg => net.k, :f => net.m, :psh => net.n, :lmp => net.n)
        # Expected sizes for each parameter
        param_sizes = Dict(:d => net.n, :sw => net.m, :cl => net.k, :cq => net.k,
                           :fmax => net.m, :b => net.m)

        # 5 operands × 6 parameters = 30 combinations

        for op in operands
            for param in params
                expected_rows = op_sizes[op]
                expected_cols = param_sizes[param]
                @testset "$op w.r.t. $param" begin
                    S = calc_sensitivity(prob, op, param)
                    @test S isa Sensitivity
                    @test S.formulation == :dcopf
                    @test size(S) == (expected_rows, expected_cols)
                    @test all(isfinite, Matrix(S))
                end
            end
        end

        # Invalid combinations
        @test_throws ArgumentError calc_sensitivity(prob, :vm, :d)
        @test_throws ArgumentError calc_sensitivity(prob, :qg, :d)
    end

    @testset "AC Power Flow — 24 native + 10 transform = 34 combinations" begin
        # Solve AC power flow first
        pf_data = deepcopy(net_data)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        # 24 native combinations
        native_combos = [
            # Existing 6
            (:vm, :p, (state.n, state.n)),
            (:vm, :q, (state.n, state.n)),
            (:v,  :p, (state.n, state.n)),
            (:v,  :q, (state.n, state.n)),
            (:im, :p, (state.m, state.n)),
            (:im, :q, (state.m, state.n)),
            # Voltage angle
            (:va, :p, (state.n, state.n)),
            (:va, :q, (state.n, state.n)),
            # Branch flow
            (:f,  :p, (state.m, state.n)),
            (:f,  :q, (state.m, state.n)),
            # Jacobian blocks
            (:p,  :va, (state.n, state.n)),
            (:p,  :vm, (state.n, state.n)),
            (:q,  :va, (state.n, state.n)),
            (:q,  :vm, (state.n, state.n)),
            # Topology: conductance
            (:vm, :g, (state.n, state.m)),
            (:va, :g, (state.n, state.m)),
            (:v,  :g, (state.n, state.m)),
            (:f,  :g, (state.m, state.m)),
            (:im, :g, (state.m, state.m)),
            # Topology: susceptance
            (:vm, :b, (state.n, state.m)),
            (:va, :b, (state.n, state.m)),
            (:v,  :b, (state.n, state.m)),
            (:f,  :b, (state.m, state.m)),
            (:im, :b, (state.m, state.m)),
        ]

        for (op, param, expected_size) in native_combos
            @testset "native: $op w.r.t. $param" begin
                S = calc_sensitivity(state, op, param)
                @test S isa Sensitivity
                @test S.formulation == :acpf
                @test size(S) == expected_size
                @test all(isfinite, Matrix(S))
            end
        end

        # 10 transform-derived combinations (:d and :qd)
        # Note: (:p,:d) and (:q,:d) are NOT valid — the Jacobian blocks (:p,:va)
        # and (:p,:vm) don't have :p as a native parameter, so the :d transform
        # (which requires native (:op,:p)) doesn't apply to :p/:q operands.
        transform_combos = [
            # 5 via :d (from :p)
            (:vm, :d, (state.n, state.n)),
            (:v,  :d, (state.n, state.n)),
            (:im, :d, (state.m, state.n)),
            (:va, :d, (state.n, state.n)),
            (:f,  :d, (state.m, state.n)),
            # 5 via :qd (from :q)
            (:vm, :qd, (state.n, state.n)),
            (:v,  :qd, (state.n, state.n)),
            (:im, :qd, (state.m, state.n)),
            (:va, :qd, (state.n, state.n)),
            (:f,  :qd, (state.m, state.n)),
        ]

        for (op, param, expected_size) in transform_combos
            @testset "transform: $op w.r.t. $param" begin
                S = calc_sensitivity(state, op, param)
                @test S isa Sensitivity
                @test S.formulation == :acpf
                @test size(S) == expected_size
                @test all(isfinite, Matrix(S))
            end
        end

        # Verify transform relationship: ∂/∂d = -∂/∂p
        dvm_dp = calc_sensitivity(state, :vm, :p)
        dvm_dd = calc_sensitivity(state, :vm, :d)
        @test Matrix(dvm_dd) ≈ -Matrix(dvm_dp)

        dvm_dq = calc_sensitivity(state, :vm, :q)
        dvm_dqd = calc_sensitivity(state, :vm, :qd)
        @test Matrix(dvm_dqd) ≈ -Matrix(dvm_dq)

        # Jacobian block + demand transform is invalid
        @test_throws ArgumentError calc_sensitivity(state, :p, :d)
        @test_throws ArgumentError calc_sensitivity(state, :q, :d)

        # Invalid combinations
        @test_throws ArgumentError calc_sensitivity(state, :lmp, :p)
        @test_throws ArgumentError calc_sensitivity(state, :pg, :p)
    end

    @testset "AC OPF — all 36 combinations" begin
        prob = ACOPFProblem(net_data; silent=true)

        n, m, k = prob.network.n, prob.network.m, prob.n_gen
        operands = [:va, :vm, :pg, :qg, :lmp, :qlmp]
        params = [:sw, :d, :qd, :cq, :cl, :fmax]

        op_sizes = Dict(:va => n, :vm => n, :pg => k, :qg => k, :lmp => n, :qlmp => n)
        param_sizes = Dict(:sw => m, :d => n, :qd => n, :cq => k, :cl => k, :fmax => m)

        for op in operands
            for param in params
                expected_rows = op_sizes[op]
                expected_cols = param_sizes[param]
                @testset "$op w.r.t. $param" begin
                    S = calc_sensitivity(prob, op, param)
                    @test S isa Sensitivity
                    @test S.formulation == :acopf
                    @test size(S) == (expected_rows, expected_cols)
                    @test all(isfinite, Matrix(S))
                end
            end
        end

        # Invalid combinations
        @test_throws ArgumentError calc_sensitivity(prob, :f, :sw)
        @test_throws ArgumentError calc_sensitivity(prob, :psh, :d)
    end

    @testset "Symbol aliases" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        solve!(prob)

        # :g is alias for :pg
        s1 = calc_sensitivity(prob, :pg, :d)
        s2 = calc_sensitivity(prob, :g, :d)
        @test Matrix(s1) == Matrix(s2)

        # :pd is alias for :d
        s3 = calc_sensitivity(prob, :va, :pd)
        s4 = calc_sensitivity(prob, :va, :d)
        @test Matrix(s3) == Matrix(s4)
    end
end
