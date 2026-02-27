# Exhaustive sensitivity coverage test
# Verifies that ALL valid (operand, parameter) combinations work for each formulation.

using PowerModelsDiff
using PowerModels
using LinearAlgebra
using Test

@testset "Sensitivity Coverage" begin
    # Load test case
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")
    pm_data = PowerModels.parse_file(file)
    net_data = PowerModels.make_basic_network(pm_data)

    @testset "DC Power Flow — all 4 combinations" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        pf = DCPowerFlowState(net, d)

        combos = [
            (:va, :d, (net.n, net.n)),
            (:f,  :d, (net.m, net.n)),
            (:va, :sw, (net.n, net.m)),
            (:f,  :sw, (net.m, net.m)),
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

    @testset "DC OPF — all 24 combinations" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        solve!(prob)

        operands = [:va, :pg, :f, :lmp]
        params = [:d, :sw, :cl, :cq, :fmax, :b]

        # Expected sizes for each operand
        op_sizes = Dict(:va => net.n, :pg => net.k, :f => net.m, :lmp => net.n)
        # Expected sizes for each parameter
        param_sizes = Dict(:d => net.n, :sw => net.m, :cl => net.k, :cq => net.k,
                           :fmax => net.m, :b => net.m)

        # 4 operands × 6 parameters = 24 combinations

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

    @testset "AC Power Flow — all 6 combinations" begin
        # Solve AC power flow first
        pf_data = deepcopy(net_data)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        combos = [
            (:vm, :p, (state.n, state.n)),
            (:vm, :q, (state.n, state.n)),
            (:v,  :p, (state.n, state.n)),
            (:v,  :q, (state.n, state.n)),
            (:im, :p, (state.m, state.n)),
            (:im, :q, (state.m, state.n)),
        ]

        for (op, param, expected_size) in combos
            @testset "$op w.r.t. $param" begin
                S = calc_sensitivity(state, op, param)
                @test S isa Sensitivity
                @test S.formulation == :acpf
                @test size(S) == expected_size
                @test all(isfinite, Matrix(S))
            end
        end

        # Invalid combinations
        @test_throws ArgumentError calc_sensitivity(state, :lmp, :p)
        @test_throws ArgumentError calc_sensitivity(state, :pg, :p)
    end

    @testset "AC OPF — all 4 combinations" begin
        prob = ACOPFProblem(net_data; silent=true)

        combos = [
            (:vm, :sw, (prob.network.n, prob.network.m)),
            (:va, :sw, (prob.network.n, prob.network.m)),
            (:pg, :sw, (prob.n_gen, prob.network.m)),
            (:qg, :sw, (prob.n_gen, prob.network.m)),
        ]

        for (op, param, expected_size) in combos
            @testset "$op w.r.t. $param" begin
                S = calc_sensitivity(prob, op, param)
                @test S isa Sensitivity
                @test S.formulation == :acopf
                @test size(S) == expected_size
                @test all(isfinite, Matrix(S))
            end
        end

        # Invalid combinations
        @test_throws ArgumentError calc_sensitivity(prob, :lmp, :sw)
        @test_throws ArgumentError calc_sensitivity(prob, :f, :sw)
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
