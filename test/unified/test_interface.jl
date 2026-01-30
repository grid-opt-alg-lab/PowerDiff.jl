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

    @testset "Singleton Type Tags" begin
        # Test formulation tags
        @test DCOPF() isa AbstractFormulation
        @test DCPF() isa AbstractFormulation
        @test ACOPF() isa AbstractFormulation
        @test ACPF() isa AbstractFormulation

        # Test operand tags
        @test VoltageAngle() isa AbstractOperand
        @test VoltageMagnitude() isa AbstractOperand
        @test LMP() isa AbstractOperand
        @test Generation() isa AbstractOperand
        @test Flow() isa AbstractOperand

        # Test parameter tags
        @test Demand() isa AbstractParameter
        @test Switching() isa AbstractParameter
        @test QuadraticCost() isa AbstractParameter
        @test LinearCost() isa AbstractParameter
        @test FlowLimit() isa AbstractParameter
        @test Susceptance() isa AbstractParameter
        @test ActivePower() isa AbstractParameter
        @test ReactivePower() isa AbstractParameter
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

        # Test type-based interface
        dva_dz = calc_sensitivity(pf_state, VoltageAngle(), Switching())
        @test dva_dz isa Sensitivity{DCPF, VoltageAngle, Switching}
        @test size(dva_dz) == (net.n, net.m)

        df_dz = calc_sensitivity(pf_state, Flow(), Switching())
        @test df_dz isa Sensitivity{DCPF, Flow, Switching}
        @test size(df_dz) == (net.m, net.m)
    end

    @testset "DC Power Flow Demand Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        pf_state = DCPowerFlowState(net, demand)

        # Test type-based interface
        dva_dd = calc_sensitivity(pf_state, VoltageAngle(), Demand())
        @test dva_dd isa Sensitivity{DCPF, VoltageAngle, Demand}
        @test size(dva_dd) == (net.n, net.n)

        df_dd = calc_sensitivity(pf_state, Flow(), Demand())
        @test df_dd isa Sensitivity{DCPF, Flow, Demand}
        @test size(df_dd) == (net.m, net.n)
    end

    @testset "DC OPF Switching Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, demand)

        # Test type-based interface
        dva_dz = calc_sensitivity(prob, VoltageAngle(), Switching())
        @test dva_dz isa Sensitivity{DCOPF, VoltageAngle, Switching}
        @test size(dva_dz) == (net.n, net.m)

        dg_dz = calc_sensitivity(prob, Generation(), Switching())
        @test dg_dz isa Sensitivity{DCOPF, Generation, Switching}
        @test size(dg_dz) == (net.k, net.m)

        df_dz = calc_sensitivity(prob, Flow(), Switching())
        @test df_dz isa Sensitivity{DCOPF, Flow, Switching}
        @test size(df_dz) == (net.m, net.m)
    end

    @testset "DC OPF Demand Sensitivity" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, demand)

        # Test type-based interface returns typed result
        dlmp_dd = calc_sensitivity(prob, LMP(), Demand())
        @test dlmp_dd isa Sensitivity{DCOPF, LMP, Demand}
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

        # Voltage-power sensitivities
        sens = calc_voltage_power_sensitivities(state)
        @test sens isa VoltagePowerSensitivity
        @test size(sens.∂vm_∂p) == (state.n, state.n)
        @test size(sens.∂vm_∂q) == (state.n, state.n)

        # Test non-unicode accessors
        @test sens.dvm_dp == sens.∂vm_∂p
        @test sens.dvm_dq == sens.∂vm_∂q

        # Type-based interface
        dvm_dp = calc_sensitivity(state, VoltageMagnitude(), ActivePower())
        @test dvm_dp isa Sensitivity{ACPF, VoltageMagnitude, ActivePower}
        @test Matrix(dvm_dp) == sens.∂vm_∂p
    end

    @testset "Sensitivity Type Dispatch" begin
        net = DCNetwork(net_data)
        demand = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, demand)

        # Get typed sensitivity
        sens = calc_sensitivity(prob, LMP(), Demand())

        # Test dispatch helpers
        @test formulation(sens) == DCOPF
        @test operand(sens) == LMP
        @test parameter(sens) == Demand

        # Test type-based dispatch
        function process_sens(s::Sensitivity{DCOPF, LMP, Demand})
            return "DC OPF LMP-demand"
        end
        function process_sens(s::Sensitivity{F, O, P}) where {F, O, P}
            return "Generic sensitivity"
        end

        @test process_sens(sens) == "DC OPF LMP-demand"

        # Test dispatching on any switching sensitivity
        dva_dz = calc_sensitivity(prob, VoltageAngle(), Switching())
        function process_switching(s::Sensitivity{F, O, Switching}) where {F, O}
            return "Switching sensitivity"
        end
        @test process_switching(dva_dz) == "Switching sensitivity"
    end
end

println("All unified architecture tests passed!")
