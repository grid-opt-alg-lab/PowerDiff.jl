using PowerModelsDiff
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
        @test length(pf_state.θ) == net.n
        @test length(pf_state.f) == net.m
        @test pf_state.p == -demand  # Since g = 0

        # Test with generation
        g = zeros(net.n)
        g[1] = sum(demand)  # All generation at bus 1
        pf_state2 = DCPowerFlowState(net, g, demand)
        @test pf_state2.g == g
        @test pf_state2.d == demand
        @test pf_state2.p == g - demand
    end

    @testset "DC Power Flow Switching Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, demand)

        dva_dz = calc_sensitivity(pf_state, :va, :z)
        @test dva_dz isa Sensitivity
        @test dva_dz.formulation == :dcpf
        @test dva_dz.operand == :va
        @test dva_dz.parameter == :z
        @test size(dva_dz) == (net.n, net.m)

        df_dz = calc_sensitivity(pf_state, :f, :z)
        @test df_dz isa Sensitivity
        @test df_dz.formulation == :dcpf
        @test df_dz.operand == :f
        @test df_dz.parameter == :z
        @test size(df_dz) == (net.m, net.m)
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

        dva_dz = calc_sensitivity(prob, :va, :z)
        @test dva_dz isa Sensitivity
        @test dva_dz.formulation == :dcopf
        @test size(dva_dz) == (net.n, net.m)

        dg_dz = calc_sensitivity(prob, :pg, :z)
        @test dg_dz isa Sensitivity
        @test dg_dz.formulation == :dcopf
        @test dg_dz.operand == :pg
        @test size(dg_dz) == (net.k, net.m)

        df_dz = calc_sensitivity(prob, :f, :z)
        @test df_dz isa Sensitivity
        @test df_dz.formulation == :dcopf
        @test size(df_dz) == (net.m, net.m)
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
        z = ones(ac_net.m)
        z[1] = 0.0  # Open first branch
        Y_switched = admittance_matrix(ac_net, z)
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
        dva_dz = calc_sensitivity(prob, :va, :z)
        @test dva_dz.formulation == :dcopf
        @test dva_dz.operand == :va
        @test dva_dz.parameter == :z
    end
end

println("All unified architecture tests passed!")
