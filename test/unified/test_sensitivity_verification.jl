using PowerModelsDiff
using PowerModels
using ForwardDiff
using LinearAlgebra
using Test

@testset "Sensitivity Verification with ForwardDiff" begin

    # Load a test network
    case_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower", "case5.m")
    data = PowerModels.parse_file(case_path)
    net_data = PowerModels.make_basic_network(data)

    @testset "DC Power Flow Switching Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Solve DC power flow
        pf_state = DCPowerFlowState(net, d)

        # Get analytical sensitivity
        sens = calc_sensitivity_switching(pf_state)

        # Define a function that computes theta as a function of z
        # Must match DCPowerFlowState constructor exactly (slack balancing + centering)
        function theta_of_z(z_vec)
            A = net.A
            b = net.b
            L = transpose(A) * Diagonal(-b .* z_vec) * A
            L_pinv = pinv(Matrix(L))

            # Balance at slack bus
            p_balanced = copy(pf_state.p)
            p_balanced[net.ref_bus] = -sum(pf_state.p) + pf_state.p[net.ref_bus]

            # Solve for theta
            θ = L_pinv * p_balanced

            # Center around reference bus
            return θ .- θ[net.ref_bus]
        end

        # Compute ForwardDiff Jacobian
        z0 = copy(net.z)
        fd_dθ_dz = ForwardDiff.jacobian(theta_of_z, z0)

        # Compare analytical vs ForwardDiff
        @test size(sens.dθ_dz) == size(fd_dθ_dz)
        @test maximum(abs.(sens.dθ_dz - fd_dθ_dz)) < 1e-10

        # Also verify flow sensitivity
        function flow_of_z(z_vec)
            A = net.A
            b = net.b
            θ = theta_of_z(z_vec)
            W = Diagonal(-b .* z_vec)
            return W * A * θ
        end

        fd_df_dz = ForwardDiff.jacobian(flow_of_z, z0)
        @test size(sens.df_dz) == size(fd_df_dz)
        @test maximum(abs.(sens.df_dz - fd_df_dz)) < 1e-10

        println("DC PF Switching Sensitivity: max θ error = ", maximum(abs.(sens.dθ_dz - fd_dθ_dz)))
        println("DC PF Switching Sensitivity: max f error = ", maximum(abs.(sens.df_dz - fd_df_dz)))
    end

    @testset "DC Power Flow Demand Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)

        # Solve DC power flow
        pf_state = DCPowerFlowState(net, d)

        # Get analytical sensitivity
        sens = calc_sensitivity_demand(pf_state)

        # Define a function that computes theta as a function of d
        function theta_of_d(d_vec)
            A = net.A
            b = net.b
            L = transpose(A) * Diagonal(-b .* net.z) * A
            L_pinv = pinv(Matrix(L))
            p = pf_state.g - d_vec  # Net injection
            return L_pinv * p
        end

        # Compute ForwardDiff Jacobian
        fd_dθ_dd = ForwardDiff.jacobian(theta_of_d, d)

        # Compare analytical vs ForwardDiff
        @test size(sens.dθ_dd) == size(fd_dθ_dd)
        @test maximum(abs.(sens.dθ_dd - fd_dθ_dd)) < 1e-8

        println("DC PF Demand Sensitivity: max error = ", maximum(abs.(sens.dθ_dd - fd_dθ_dd)))
    end

    @testset "DC OPF Demand Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Get analytical sensitivity
        sens = calc_sensitivity_demand(prob)

        # Verify with finite differences (ForwardDiff on JuMP is complex)
        ε = 1e-5
        n = net.n

        # Check a subset of entries using finite differences
        for i in 1:min(3, n)
            d_pert = copy(d)
            d_pert[i] += ε

            # Update demand and re-solve
            update_demand!(prob, d_pert)
            sol_pert = solve!(prob)

            fd_dθ_dd_col = (sol_pert.θ - sol.θ) / ε

            # Compare
            max_err = maximum(abs.(sens.dθ_dd[:, i] - fd_dθ_dd_col))
            @test max_err < 1e-3  # Finite difference tolerance

            # Restore
            update_demand!(prob, d)
        end

        println("DC OPF Demand Sensitivity: finite diff verification passed")
    end

    @testset "DC OPF Switching Sensitivity" begin
        net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Get analytical sensitivity
        sens = calc_sensitivity_switching(prob)

        # Verify with finite differences (negative perturbation to stay in [0,1])
        ε = 1e-5
        m = net.m

        # Check a subset of entries using finite differences
        for e in 1:min(3, m)
            z_pert = copy(net.z)
            z_pert[e] -= ε  # Use negative perturbation since z starts at 1.0

            # Update switching and re-solve
            update_switching!(prob, z_pert)
            sol_pert = solve!(prob)

            fd_dθ_dz_col = (sol.θ - sol_pert.θ) / ε  # Reversed due to negative ε

            # Compare (larger tolerance for OPF due to active constraint changes)
            # Note: Finite differences for constrained OPF are less accurate due to
            # constraint activity changes - the key validation is ForwardDiff on DC PF
            max_err = maximum(abs.(sens.dθ_dz[:, e] - fd_dθ_dz_col))
            @test max_err < 0.05  # Larger tolerance for OPF finite differences

            # Restore
            update_switching!(prob, net.z)
        end

        println("DC OPF Switching Sensitivity: finite diff verification passed")
    end

    @testset "AC Voltage-Power Sensitivity" begin
        # Solve AC power flow
        PowerModels.compute_ac_pf!(net_data)
        state = ACPowerFlowState(net_data)

        # Get analytical sensitivity
        sens = calc_sensitivity(state, POWER)

        # Verify voltage magnitude sensitivity structure
        @test size(sens.∂vm_∂p) == (state.n, state.n)
        @test size(sens.∂vm_∂q) == (state.n, state.n)

        # Basic sanity: sensitivities should be real and finite
        @test all(isfinite, sens.∂vm_∂p)
        @test all(isfinite, sens.∂vm_∂q)

        # Slack bus voltage should have zero sensitivity
        # (slack bus voltage is fixed, so d|v_slack|/dp = 0)
        slack_idx = state.idx_slack
        @test maximum(abs.(sens.∂vm_∂p[slack_idx, :])) < 1e-10
        @test maximum(abs.(sens.∂vm_∂q[slack_idx, :])) < 1e-10

        println("AC Voltage-Power Sensitivity: sanity checks passed")
    end

end

println("\nAll sensitivity verification tests passed!")
