# Test AC OPF switching sensitivity

using PowerModelsDiff
using PowerModels
using Test

@testset "AC OPF Switching Sensitivity" begin
    # Load test case
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")

    pm_data = PowerModels.parse_file(file)
    pm_data = PowerModels.make_basic_network(pm_data)

    # Create and solve AC OPF
    @testset "ACOPFProblem construction and solving" begin
        prob = ACOPFProblem(pm_data; silent=true)

        @test prob.network.n == 5
        @test prob.network.m == 7
        @test prob.n_gen == 5

        sol = solve!(prob)

        @test sol.objective > 0
        @test length(sol.vm) == 5
        @test length(sol.va) == 5
        @test length(sol.pg) == 5
        @test length(sol.qg) == 5

        # Voltage magnitudes should be within limits
        @test all(sol.vm .>= 0.9)
        @test all(sol.vm .<= 1.1)

        println("AC OPF solved successfully, objective = $(round(sol.objective, digits=2))")
    end

    @testset "Switching sensitivity computation" begin
        prob = ACOPFProblem(pm_data; silent=true)
        sens = PowerModelsDiff.calc_sensitivity_switching(prob)

        @test sens isa ACOPFSwitchingSens
        @test size(sens.dvm_dz) == (5, 7)
        @test size(sens.dva_dz) == (5, 7)
        @test size(sens.dpg_dz) == (5, 7)
        @test size(sens.dqg_dz) == (5, 7)

        # Sensitivities should be finite
        @test all(isfinite.(sens.dvm_dz))
        @test all(isfinite.(sens.dva_dz))
        @test all(isfinite.(sens.dpg_dz))
        @test all(isfinite.(sens.dqg_dz))

        println("Switching sensitivities computed successfully")
        println("Sample dvm/dz[1,:] = $(round.(sens.dvm_dz[1, :], digits=4))")
    end

    @testset "Symbol-based API" begin
        prob = ACOPFProblem(pm_data; silent=true)

        dvm_dz = calc_sensitivity(prob, :vm, :z)
        @test size(dvm_dz) == (5, 7)

        dva_dz = calc_sensitivity(prob, :va, :z)
        @test size(dva_dz) == (5, 7)

        dpg_dz = calc_sensitivity(prob, :pg, :z)
        @test size(dpg_dz) == (5, 7)

        dqg_dz = calc_sensitivity(prob, :qg, :z)
        @test size(dqg_dz) == (5, 7)

        println("Symbol-based API works correctly")
    end
end

println("\n✓ All AC OPF sensitivity tests passed!")
