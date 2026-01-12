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

        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        sol = solve!(prob)
        @test sol isa AbstractOPFSolution
        @test sol isa AbstractPowerFlowState

        pf_state = DCPowerFlowState(net, d)
        @test pf_state isa AbstractPowerFlowState
    end

    @testset "Parameter Types" begin
        @test DEMAND isa DemandParameter
        @test SWITCHING isa SwitchingParameter
        @test POWER isa PowerInjectionParameter
        @test TOPOLOGY isa TopologyParameter
        @test COST isa CostParameter
    end

    @testset "DC Power Flow State" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Test construction
        pf_state = DCPowerFlowState(net, d)
        @test length(pf_state.θ) == net.n
        @test length(pf_state.f) == net.m
        @test pf_state.p == -d  # Since g = 0

        # Test with generation
        g = zeros(net.n)
        g[1] = sum(d)  # All generation at bus 1
        pf_state2 = DCPowerFlowState(net, g, d)
        @test pf_state2.g == g
        @test pf_state2.d == d
        @test pf_state2.p == g - d
    end

    @testset "DC Power Flow Switching Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, d)

        sens = calc_sensitivity_switching(pf_state)
        @test sens isa DCPFSwitchingSens
        @test size(sens.dva_dz) == (net.n, net.m)
        @test size(sens.df_dz) == (net.m, net.m)

        # Test symbol-based interface
        dva_dz = calc_sensitivity(pf_state, :va, :z)
        @test dva_dz == sens.dva_dz
    end

    @testset "DC Power Flow Demand Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, d)

        sens = calc_sensitivity_demand(pf_state)
        @test sens isa DCPFDemandSens
        @test size(sens.dva_dd) == (net.n, net.n)
        @test size(sens.df_dd) == (net.m, net.n)

        # Test symbol-based interface
        dva_dd = calc_sensitivity(pf_state, :va, :d)
        @test dva_dd == sens.dva_dd
    end

    @testset "DC OPF Switching Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)

        sens = calc_sensitivity_switching(prob)
        @test sens isa OPFSwitchingSens
        @test size(sens.dva_dz) == (net.n, net.m)
        @test size(sens.dg_dz) == (net.k, net.m)
        @test size(sens.df_dz) == (net.m, net.m)

        # Test symbol-based interface
        dva_dz = calc_sensitivity(prob, :va, :z)
        @test dva_dz == sens.dva_dz
    end

    @testset "Unified Voltage Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, d)

        # DC Power Flow
        dθ_dd = calc_voltage_sensitivity(pf_state, DEMAND)
        @test size(dθ_dd) == (net.n, net.n)

        dθ_dz = calc_voltage_sensitivity(pf_state, SWITCHING)
        @test size(dθ_dz) == (net.n, net.m)

        # DC OPF
        prob = DCOPFProblem(net, d)
        dθ_dd_opf = calc_voltage_sensitivity(prob, DEMAND)
        @test size(dθ_dd_opf) == (net.n, net.n)
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

        # Voltage-power sensitivities
        sens = calc_voltage_power_sensitivities(state)
        @test sens isa VoltagePowerSensitivity
        @test size(sens.∂vm_∂p) == (state.n, state.n)
        @test size(sens.∂vm_∂q) == (state.n, state.n)

        # Unified interface
        sens2 = calc_sensitivity(state, POWER)
        @test sens2 isa VoltagePowerSensitivity
    end
end

println("All unified architecture tests passed!")
