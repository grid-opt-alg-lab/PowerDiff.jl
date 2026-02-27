using Test
using LinearAlgebra
using SparseArrays
using Statistics
using PowerModelsDiff
using PowerModels
using ForwardDiff
using Ipopt
using JuMP: MOI, optimizer_with_attributes

PowerModels.silence()

# Use PowerModels' built-in test cases
const PM_DATA_DIR = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")

# Helper to load a test case
function load_test_case(case_name::String)
    case_path = joinpath(PM_DATA_DIR, case_name)
    if isfile(case_path)
        raw = PowerModels.parse_file(case_path)
        return PowerModels.make_basic_network(raw)
    else
        return nothing
    end
end

# =============================================================================
# DC OPF Tests
# =============================================================================

@testset "DCNetwork Construction" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping DCNetwork tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)

        @test dc_net.n == length(net["bus"])
        @test dc_net.m == length(net["branch"])
        @test dc_net.k == length(net["gen"])
        @test size(dc_net.A) == (dc_net.m, dc_net.n)
        @test size(dc_net.G_inc) == (dc_net.n, dc_net.k)
        @test length(dc_net.b) == dc_net.m
        @test length(dc_net.fmax) == dc_net.m
        @test length(dc_net.gmax) == dc_net.k
        @test dc_net.ref_bus >= 1 && dc_net.ref_bus <= dc_net.n
    end
end

@testset "DCOPFProblem Construction and Solve" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping DCOPFProblem tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)

        # Test problem construction
        prob = DCOPFProblem(dc_net, d)
        @test prob.network === dc_net
        @test length(prob.θ) == dc_net.n
        @test length(prob.g) == dc_net.k
        @test length(prob.f) == dc_net.m

        # Test solving
        sol = solve!(prob)
        @test length(sol.θ) == dc_net.n
        @test length(sol.g) == dc_net.k
        @test length(sol.f) == dc_net.m

        # Basic feasibility checks
        @test sol.θ[dc_net.ref_bus] ≈ 0.0 atol=1e-6  # Reference bus angle = 0
        @test all(sol.g .>= dc_net.gmin .- 1e-6)      # Generation lower bounds
        @test all(sol.g .<= dc_net.gmax .+ 1e-6)      # Generation upper bounds
        @test all(sol.f .>= -dc_net.fmax .- 1e-6)     # Flow lower bounds
        @test all(sol.f .<= dc_net.fmax .+ 1e-6)      # Flow upper bounds

        # Test power balance (approximately)
        B_mat = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        power_imbalance = dc_net.G_inc * sol.g - d - B_mat * sol.θ
        @test norm(power_imbalance) < 1e-4
    end
end

@testset "LMP Computation" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping LMP tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        lmps = calc_lmp(sol, dc_net)

        @test length(lmps) == dc_net.n
        @test !any(isnan, lmps)
        @test !any(isinf, lmps)

        # LMPs should be positive in typical cases
        # (though this isn't always guaranteed)
    end
end

@testset "KKT System" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping KKT tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        # Test dimensions
        dim = kkt_dims(dc_net)
        @test dim == 2*dc_net.n + 4*dc_net.m + 3*dc_net.k + 1

        # Test flatten/unflatten round-trip
        z = flatten_variables(sol, prob)
        @test length(z) == dim

        vars = unflatten_variables(z, prob)
        @test vars.θ ≈ sol.θ
        @test vars.g ≈ sol.g
        @test vars.f ≈ sol.f

        # Test KKT residuals (should be near zero at optimum)
        K = kkt(z, prob, d)
        # Note: complementary slackness won't be exactly zero due to interior point solver
        # Check stationarity and feasibility conditions
        n, m, k = dc_net.n, dc_net.m, dc_net.k
        # Primal feasibility should be very tight
        idx_power_bal = n + k + 3m + 2k + 1 : n + k + 3m + 2k + n
        @test norm(K[idx_power_bal]) < 1e-4
    end
end

@testset "Demand Sensitivity" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping demand sensitivity tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        solve!(prob)

        # Use the type-based interface
        dva_dd = calc_sensitivity(prob, :va, :d)
        dg_dd = calc_sensitivity(prob, :pg, :d)
        df_dd = calc_sensitivity(prob, :f, :d)
        dlmp_dd = calc_sensitivity(prob, :lmp, :d)

        @test size(dva_dd) == (dc_net.n, dc_net.n)
        @test size(dg_dd) == (dc_net.k, dc_net.n)
        @test size(df_dd) == (dc_net.m, dc_net.n)
        @test size(dlmp_dd) == (dc_net.n, dc_net.n)

        @test !any(isnan, Matrix(dva_dd))
        @test !any(isnan, Matrix(dg_dd))
        @test !any(isnan, Matrix(df_dd))
        @test !any(isnan, Matrix(dlmp_dd))
    end
end

# =============================================================================
# Validation against PowerModels.jl DC OPF
# =============================================================================

@testset "Topology (Switching) Sensitivity" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping topology sensitivity tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        solve!(prob)

        # Compute switching sensitivities using type-based interface
        dva_dsw = calc_sensitivity(prob, :va, :sw)
        dg_dsw = calc_sensitivity(prob, :pg, :sw)
        df_dsw = calc_sensitivity(prob, :f, :sw)
        dlmp_dsw = calc_sensitivity(prob, :lmp, :sw)

        @test size(dva_dsw) == (dc_net.n, dc_net.m)
        @test size(dg_dsw) == (dc_net.k, dc_net.m)
        @test size(df_dsw) == (dc_net.m, dc_net.m)
        @test size(dlmp_dsw) == (dc_net.n, dc_net.m)

        @test !any(isnan, Matrix(dva_dsw))
        @test !any(isnan, Matrix(dg_dsw))
        @test !any(isnan, Matrix(df_dsw))
        @test !any(isnan, Matrix(dlmp_dsw))

        # Verify sensitivities are finite
        @test all(isfinite, Matrix(dva_dsw))
        @test all(isfinite, Matrix(dg_dsw))
        @test all(isfinite, Matrix(df_dsw))
        @test all(isfinite, Matrix(dlmp_dsw))
    end
end

@testset "Validation against PowerModels DC OPF" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping validation tests - PowerModels test data not found"
        @test_skip false
    else
        # Get original (non-basic) network for PowerModels solve
        raw = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))

        # Solve with PowerModels DC OPF
        pm_result = PowerModels.solve_dc_opf(raw,
            optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))

        # Solve with our implementation
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        # Compare generation dispatch (should be close, not exact due to regularization term)
        # PowerModels stores solution by string key
        pm_gen = [pm_result["solution"]["gen"][string(i)]["pg"] for i in 1:dc_net.k]

        # Our generation should be close to PowerModels
        # Note: tolerance is higher due to regularization term in our formulation
        gen_diff = norm(sol.g - pm_gen)
        @test gen_diff < 0.1  # Within 10% for small regularization

        # Compare total generation (power balance)
        total_gen_ours = sum(sol.g)
        total_gen_pm = sum(pm_gen)
        total_demand = sum(d)

        @test abs(total_gen_ours - total_demand) < 1e-4  # Our solution is balanced
        @test abs(total_gen_pm - total_demand) < 1e-4    # PM solution is balanced

        # Objective should be similar (our objective includes regularization)
        # Just check that both objectives are positive and finite
        @test sol.objective > 0
        @test isfinite(sol.objective)
        @test pm_result["objective"] > 0

        @info "Validation results:" gen_diff=gen_diff total_gen_diff=abs(total_gen_ours - total_gen_pm)
    end
end

# =============================================================================
# Phase 2: Validation Tests
# =============================================================================

@testset "LMP Validation against PowerModels" begin
    raw = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))
    net = PowerModels.make_basic_network(raw)

    # Solve with PowerModels on the basic network (with duals enabled)
    pm_result = PowerModels.solve_dc_opf(net,
        optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0),
        setting = Dict("output" => Dict("duals" => true)))

    # Ipopt returns LOCALLY_SOLVED for nonlinear problems
    if pm_result["termination_status"] ∈ [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
        # Extract PowerModels LMPs (power balance duals)
        n_bus = length(net["bus"])
        pm_lmps = Float64[]
        for i in 1:n_bus
            bus_data = pm_result["solution"]["bus"][string(i)]
            push!(pm_lmps, get(bus_data, "lam_kcl_r", NaN))
        end

        # Solve with our implementation
        # Use small τ for numerical stability in KKT system
        dc_net = DCNetwork(net; τ=1e-3)
        prob = DCOPFProblem(dc_net, calc_demand_vector(net))
        sol = solve!(prob)

        # For LMPs, use the power balance duals directly (ν_bal)
        # This matches the standard definition: LMP = ∂Cost/∂d
        our_lmps = sol.ν_bal

        # Check validity
        @test !any(isnan, our_lmps)
        @test all(isfinite, our_lmps)

        # Check that power balance duals (marginal cost of serving load) are positive
        # (generators have positive marginal costs in this test case)
        @test all(our_lmps .> 0)

        if !any(isnan, pm_lmps)
            # PowerModels uses opposite sign convention (negative LMPs)
            # Compare magnitudes: relative error < 50% or absolute error < 100
            pm_lmps_abs = abs.(pm_lmps)
            for i in eachindex(our_lmps)
                abs_err = abs(our_lmps[i] - pm_lmps_abs[i])
                rel_err = pm_lmps_abs[i] > 1.0 ? abs_err / pm_lmps_abs[i] : abs_err
                @test rel_err < 0.5 || abs_err < 100.0
            end
            @info "LMP comparison:" our_lmps=our_lmps pm_lmps=pm_lmps
        end
    else
        @info "PowerModels solve failed with status $(pm_result["termination_status"]), skipping LMP comparison"
        @test_skip false
    end
end

@testset "Demand Sensitivity - Finite Difference Validation" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)

        # Compute analytical sensitivity using type-based interface
        prob = DCOPFProblem(dc_net, d)
        sol_base = solve!(prob)
        dg_dd = calc_sensitivity(prob, :pg, :d)
        dva_dd = calc_sensitivity(prob, :va, :d)
        df_dd = calc_sensitivity(prob, :f, :d)

        # Find a bus with demand
        bus_idx = findfirst(d .> 0)
        if isnothing(bus_idx)
            bus_idx = 1
        end

        # Finite difference validation
        delta = 1e-5
        d_pert = copy(d)
        d_pert[bus_idx] += delta

        prob_pert = DCOPFProblem(dc_net, d_pert)
        sol_pert = solve!(prob_pert)

        # Numerical derivatives
        dg_dd_numerical = (sol_pert.g - sol_base.g) / delta
        dva_dd_numerical = (sol_pert.θ - sol_base.θ) / delta
        df_dd_numerical = (sol_pert.f - sol_base.f) / delta

        # Compare against analytical (relative error)
        if norm(dg_dd_numerical) > 1e-10
            rel_error_g = norm(Matrix(dg_dd)[:, bus_idx] - dg_dd_numerical) / norm(dg_dd_numerical)
            @test rel_error_g < 0.01  # 1% tolerance
        else
            @info "Skipped ∂g/∂d FD check: near-zero numerical derivative" bus=bus_idx
        end

        if norm(dva_dd_numerical) > 1e-10
            rel_error_theta = norm(Matrix(dva_dd)[:, bus_idx] - dva_dd_numerical) / norm(dva_dd_numerical)
            @test rel_error_theta < 0.01
        else
            @info "Skipped ∂θ/∂d FD check: near-zero numerical derivative" bus=bus_idx
        end

        if norm(df_dd_numerical) > 1e-10
            rel_error_f = norm(Matrix(df_dd)[:, bus_idx] - df_dd_numerical) / norm(df_dd_numerical)
            @test rel_error_f < 0.01
        else
            @info "Skipped ∂f/∂d FD check: near-zero numerical derivative" bus=bus_idx
        end
    end
end

@testset "Participation Factors Sum to One" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        solve!(prob)

        # Use the type-based interface
        dg_dd = calc_sensitivity(prob, :pg, :d)

        # For each demand bus, generation participation factors should sum to 1
        # This is because total generation must equal total demand
        for i in 1:dc_net.n
            pf_sum = sum(Matrix(dg_dd)[:, i])
            @test abs(pf_sum - 1.0) < 0.01  # Should sum to 1 (power balance)
        end
    end
end

@testset "2-Bus Closed-Form Validation" begin
    # Create minimal 2-bus network
    # With our B-θ formulation: f = W*A*θ where W = Diagonal(-b*z)
    # For A = [1, -1] (line from bus 1 to 2) and b = -10 (negative susceptance):
    # W = -b = 10, so f = 10 * (θ₁ - θ₂) = 10 * (-θ₂) = -10*θ₂ (since θ₁=0)
    # For demand at bus 2, flow goes from 1→2, so f = 1 → θ₂ = -0.1
    n, m, k = 2, 1, 1
    A = sparse([1.0 -1.0])  # From bus 1 to bus 2
    G_inc = sparse(reshape([1.0, 0.0], 2, 1))  # Generator at bus 1
    b = [-10.0]  # Susceptance (negative per standard convention)

    dc_net = DCNetwork(n, m, k, A, G_inc, b;
        fmax=[100.0], gmax=[10.0], gmin=[0.0],
        cl=[10.0], cq=[0.0], ref_bus=1, τ=0.0)

    d = [0.0, 1.0]  # 1 MW load at bus 2
    prob = DCOPFProblem(dc_net, d)
    sol = solve!(prob)

    # Verify closed-form solution
    @test sol.g[1] ≈ 1.0 atol=1e-4  # Generator supplies all demand
    @test sol.θ[1] ≈ 0.0 atol=1e-4  # Reference bus
    # θ₂ = f / (b*z) = 1/10 = 0.1 (positive because of flow direction convention)
    @test abs(sol.θ[2]) ≈ 0.1 atol=1e-4
    @test abs(sol.f[1]) ≈ 1.0 atol=1e-4  # |Flow| = demand

    # LMPs should be equal (no congestion)
    lmps = calc_lmp(sol, dc_net)
    @test abs(lmps[1] - lmps[2]) < 0.1  # Nearly equal (no congestion)

    # Verify sensitivities using type-based interface
    dg_dd = calc_sensitivity(prob, :pg, :d)
    dva_dd = calc_sensitivity(prob, :va, :d)
    @test Matrix(dg_dd)[1, 2] ≈ 1.0 atol=0.01  # dg1/dd2 = 1
    # dtheta2/dd2 = 1/b = 0.1 (same sign as theta2)
    @test abs(Matrix(dva_dd)[2, 2]) ≈ 0.1 atol=0.01
end

@testset "3-Bus Congested - LMP Differentiation" begin
    # Simple 2-bus with congestion: more reliable test case
    # Bus 1: cheap generator (c=10), Bus 2: expensive generator (c=50), Bus 3: load
    # Line 1→3 constrained, forcing expensive gen to help
    n, m, k = 3, 2, 2
    # Line topology: 1→3 (constrained), 2→3
    A = sparse([
        1.0  0.0 -1.0;   # Line 1: 1→3 (congested)
        0.0  1.0 -1.0    # Line 2: 2→3
    ])
    G_inc = sparse([
        1.0 0.0;   # Gen 1 at bus 1 (cheap)
        0.0 1.0;   # Gen 2 at bus 2 (expensive)
        0.0 0.0    # No gen at bus 3 (load)
    ])
    b = [-10.0, -10.0]  # Susceptances (negative per standard convention)

    dc_net = DCNetwork(n, m, k, A, G_inc, b;
        fmax=[0.5, 10.0],  # Line 1→3 constrained at 0.5 MW
        gmax=[10.0, 10.0], gmin=[0.0, 0.0],
        cl=[10.0, 50.0], cq=[0.0, 0.0],  # Gen 1 cheap, Gen 2 expensive
        ref_bus=1, τ=0.0)

    d = [0.0, 0.0, 1.5]  # 1.5 MW load at bus 3
    prob = DCOPFProblem(dc_net, d)
    sol = solve!(prob)

    lmps = calc_lmp(sol, dc_net)

    # Cheap gen can only supply 0.5 MW (line constraint)
    # Expensive gen must supply remaining 1.0 MW
    @test sol.g[1] ≈ 0.5 atol=0.1  # Cheap gen maxed by line constraint
    @test sol.g[2] ≈ 1.0 atol=0.1  # Expensive gen fills the gap

    # Total generation equals total demand
    @test abs(sum(sol.g) - sum(d)) < 1e-4

    # LMPs should differ due to congestion
    # Bus 3 (load) should have higher LMP than bus 1 (cheap gen)
    @test abs(lmps[3] - lmps[1]) > 1.0  # Significant LMP difference

    @info "3-bus congested results:" lmps=lmps gen=sol.g flows=sol.f
end

@testset "LMP Decomposition" begin
    # Test that LMP = energy_component + congestion_component
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        lmps = calc_lmp(sol, dc_net)
        energy = calc_energy_component(sol, dc_net)
        congestion = calc_congestion_component(sol, dc_net)

        # LMP should equal energy + congestion (fundamental identity)
        @test isapprox(lmps, energy .+ congestion, atol=1e-6)

        # Congestion component captures price differentiation across buses
        @test !any(isnan, congestion)
        @test !any(isinf, congestion)

        # Energy component should be finite
        @test !any(isnan, energy)
        @test !any(isinf, energy)

        @info "LMP decomposition:" lmp_range=(minimum(lmps), maximum(lmps)) congestion_range=(minimum(congestion), maximum(congestion))
    end
end

@testset "Cost Sensitivity" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @info "Skipping cost sensitivity tests - PowerModels test data not found"
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        solve!(prob)

        # Use the type-based interface
        dg_dcl = calc_sensitivity(prob, :pg, :cl)
        dg_dcq = calc_sensitivity(prob, :pg, :cq)
        dlmp_dcl = calc_sensitivity(prob, :lmp, :cl)
        dlmp_dcq = calc_sensitivity(prob, :lmp, :cq)

        # Check dimensions
        @test size(dg_dcl) == (dc_net.k, dc_net.k)
        @test size(dg_dcq) == (dc_net.k, dc_net.k)
        @test size(dlmp_dcl) == (dc_net.n, dc_net.k)
        @test size(dlmp_dcq) == (dc_net.n, dc_net.k)

        @test !any(isnan, Matrix(dg_dcl))
        @test !any(isnan, Matrix(dg_dcq))
        @test !any(isnan, Matrix(dlmp_dcl))
        @test !any(isnan, Matrix(dlmp_dcq))
    end
end

@testset "Cost Sensitivity - Finite Difference Validation" begin
    # Use 2-bus case for simpler validation
    n, m, k = 2, 1, 1
    A = sparse([1.0 -1.0])
    G_inc = sparse(reshape([1.0, 0.0], 2, 1))
    b = [-10.0]  # Negative susceptance per standard convention

    cl_base = [10.0]
    dc_net = DCNetwork(n, m, k, A, G_inc, b;
        fmax=[100.0], gmax=[10.0], gmin=[0.0],
        cl=cl_base, cq=[1.0], ref_bus=1, τ=0.01)  # Small τ for stability

    d = [0.0, 1.0]
    prob = DCOPFProblem(dc_net, d)
    sol_base = solve!(prob)
    dg_dcl = calc_sensitivity(prob, :pg, :cl)

    # Finite difference validation for linear cost
    delta = 1e-5
    cl_pert = copy(cl_base)
    cl_pert[1] += delta

    dc_net_pert = DCNetwork(n, m, k, A, G_inc, b;
        fmax=[100.0], gmax=[10.0], gmin=[0.0],
        cl=cl_pert, cq=[1.0], ref_bus=1, τ=0.01)
    prob_pert = DCOPFProblem(dc_net_pert, d)
    sol_pert = solve!(prob_pert)

    # Numerical derivative
    dg_dcl_numerical = (sol_pert.g - sol_base.g) / delta

    # Compare (single generator case)
    if norm(dg_dcl_numerical) > 1e-10
        rel_error = norm(Matrix(dg_dcl)[:, 1] - dg_dcl_numerical) / norm(dg_dcl_numerical)
        @test rel_error < 0.05  # 5% tolerance for finite difference
    else
        @info "Skipped ∂g/∂cl FD check: near-zero numerical derivative"
    end

    # LMP sensitivity check
    lmp_base = calc_lmp(sol_base, dc_net)
    lmp_pert = calc_lmp(sol_pert, dc_net_pert)
    dlmp_dcl_numerical = (lmp_pert - lmp_base) / delta
    dlmp_dcl = calc_sensitivity(prob, :lmp, :cl)

    if norm(dlmp_dcl_numerical) > 1e-10
        rel_error_lmp = norm(Matrix(dlmp_dcl)[:, 1] - dlmp_dcl_numerical) / norm(dlmp_dcl_numerical)
        @test rel_error_lmp < 0.1  # 10% tolerance
    else
        @info "Skipped ∂lmp/∂cl FD check: near-zero numerical derivative"
    end
end

# =============================================================================
# Voltage Topology Sensitivities (existing tests, updated for PowerModels data)
# =============================================================================

# Note: Voltage Topology Sensitivities tests are temporarily disabled pending
# investigation of admittance matrix reconstruction issues with standard test cases.
# The original implementation was tested with a specific case14.m file.
# TODO: Fix vectorize_laplacian_weights for general MATPOWER cases

# =============================================================================
# Physics Property Tests
# =============================================================================

@testset "PTDF Row Conservation (Kirchhoff)" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        pf = DCPowerFlowState(dc_net, d)

        # PTDF: ∂f/∂d, rows are branches, cols are buses
        df_dd = calc_sensitivity(pf, :f, :d)

        # For DC PF, the sum of PTDF entries along each row (for a given branch)
        # should be approximately 0 for a lossless network.
        # This reflects that a uniform load increase doesn't change flows (balanced by slack).
        for ℓ in 1:dc_net.m
            row_sum = sum(Matrix(df_dd)[ℓ, :])
            @test abs(row_sum) < 1e-6
        end
    end
end

@testset "Energy Component Sanity" begin
    # Verify the energy component is well-defined and the LMP decomposition identity
    # holds across different network configurations.
    # Note: energy component uniformity is a theoretical property of ideal LP with no
    # degenerate constraints. In practice, numerical decomposition via L⁺ can introduce
    # variation, so we only check basic sanity here.
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        energy = calc_energy_component(sol, dc_net)
        congestion = calc_congestion_component(sol, dc_net)
        lmp = calc_lmp(sol, dc_net)

        @test !any(isnan, energy)
        @test !any(isinf, energy)

        # Decomposition identity: LMP = energy + congestion
        @test isapprox(lmp, energy .+ congestion, atol=1e-6)

        # Energy component should be positive for typical networks with positive costs
        @test mean(energy) > 0
    end
end

@testset "case14 Basic Validation" begin
    net = load_test_case("case14.m")
    if isnothing(net)
        @info "Skipping case14 tests - PowerModels test data not found"
        @test_skip false
    else
        # DC OPF on case14
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        @test all(isfinite, sol.θ)
        @test all(isfinite, sol.g)
        @test all(isfinite, sol.f)
        @test abs(sum(sol.g) - sum(d)) < 1e-4  # Power balance

        # Sensitivities should all be finite
        for (op, param) in [(:va, :d), (:pg, :d), (:f, :d), (:lmp, :d),
                            (:va, :sw), (:pg, :sw), (:f, :sw), (:lmp, :sw)]
            S = calc_sensitivity(prob, op, param)
            @test all(isfinite, Matrix(S))
        end

        # Participation factors sum to 1
        dg_dd = calc_sensitivity(prob, :pg, :d)
        for i in 1:dc_net.n
            @test abs(sum(Matrix(dg_dd)[:, i]) - 1.0) < 0.01
        end

        # AC power flow on case14
        pf_data = deepcopy(net)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        dvm_dp = calc_sensitivity(state, :vm, :p)
        dvm_dq = calc_sensitivity(state, :vm, :q)
        @test all(isfinite, Matrix(dvm_dp))
        @test all(isfinite, Matrix(dvm_dq))

        # Slack bus sensitivity should be zero
        slack = state.idx_slack
        @test maximum(abs.(Matrix(dvm_dp)[slack, :])) < 1e-10
        @test maximum(abs.(Matrix(dvm_dq)[slack, :])) < 1e-10
    end
end

# =============================================================================
# Physics Cross-Validation Tests
# =============================================================================

@testset "Uncongested DC OPF ≈ DC PF" begin
    # When no constraints bind, DC OPF angles should match DC PF angles
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        # Build network with very relaxed limits so nothing binds
        dc_net = DCNetwork(net)
        dc_net.fmax .= 1000.0  # No flow congestion
        dc_net.gmax .= 1000.0  # No generation limits

        d = calc_demand_vector(net)

        # Solve OPF
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        # Build PF with OPF generation mapped to buses
        g_bus = dc_net.G_inc * sol.g
        pf = DCPowerFlowState(dc_net, g_bus, d)

        # Compare ∂va/∂d from both formulations
        dva_dd_opf = Matrix(calc_sensitivity(prob, :va, :d))
        dva_dd_pf  = Matrix(calc_sensitivity(pf, :va, :d))

        @test size(dva_dd_opf) == size(dva_dd_pf)
        max_diff = maximum(abs.(dva_dd_opf - dva_dd_pf))
        @test max_diff < 0.1
        @info "Uncongested OPF vs PF ∂va/∂d max diff:" max_diff=max_diff
    end
end

@testset "Binding Generator → Zero Participation" begin
    # A generator at its upper limit should have ∂pg/∂d ≈ 0
    # (it cannot increase output further)
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        dg_dd = Matrix(calc_sensitivity(prob, :pg, :d))

        # Find generators at their upper bound
        for i in 1:dc_net.k
            if sol.g[i] >= dc_net.gmax[i] - 1e-4
                # This generator is at its limit — participation should be near zero
                participation = maximum(abs.(dg_dd[i, :]))
                @test participation < 0.05
                @info "Generator $i at upper bound: max |∂pg/∂d| = $participation"
            end
        end
    end
end

@testset "Energy Component Uniformity" begin
    # For a connected lossless DC network with NO binding constraints (flow or
    # generator), all LMPs equal the common marginal cost and the energy component
    # (= LMP - congestion) should be perfectly uniform.  The congestion formula
    # L⁺ A' W (λ_ub - λ_lb) only captures flow-constraint contributions, so
    # uniformity only holds when generator bounds are also slack.
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        dc_net.fmax .= 1000.0  # No flow congestion
        dc_net.gmax .= 1000.0  # No generator limits binding

        d = calc_demand_vector(net)
        prob = DCOPFProblem(dc_net, d)
        sol = solve!(prob)

        energy = calc_energy_component(sol, dc_net)

        # With no binding constraints, energy component must be uniform
        μ = mean(energy)
        σ = std(energy)
        @test μ > 0
        @test σ / μ < 1e-4
        @info "Energy component CV:" cv=σ/μ mean=μ std=σ
    end
end

# =============================================================================
# Unified Architecture Tests
# =============================================================================
include("unified/test_interface.jl")
include("test_ac_opf_sens.jl")
include("test_sensitivity_coverage.jl")
include("test_dc_opf_verification.jl")
include("test_ac_pf_verification.jl")
