# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

function _make_bridge_network()
    A = sparse([
        1.0 -1.0  0.0  0.0
        0.0  1.0 -1.0  0.0
        0.0  0.0  1.0 -1.0
    ])
    G_inc = sparse([
        1.0 0.0
        0.0 0.0
        0.0 1.0
        0.0 0.0
    ])
    return DCNetwork(4, 3, 2, A, G_inc, fill(-10.0, 3);
        fmax=fill(10.0, 3), gmax=fill(10.0, 2), gmin=zeros(2),
        cq=fill(1.0, 2), cl=[10.0, 20.0], ref_bus=1, tau=1e-3)
end

function _make_isolated_load_network()
    A = sparse([1.0 -1.0 0.0])
    G_inc = sparse(reshape([1.0, 0.0, 0.0], 3, 1))
    return DCNetwork(3, 1, 1, A, G_inc, [-10.0];
        fmax=[10.0], gmax=[10.0], gmin=[0.0],
        cq=[1.0], cl=[10.0], ref_bus=1, tau=1e-3)
end

function _make_fully_isolated_network()
    return DCNetwork(2, 0, 2, spzeros(0, 2), sparse(Matrix(1.0I, 2, 2)), Float64[];
        fmax=Float64[], gmax=fill(10.0, 2), gmin=zeros(2),
        cq=fill(1.0, 2), cl=[10.0, 20.0], ref_bus=1, tau=1e-3)
end

@testset "Multi-island DC support" begin
    @testset "isolated load shedding and LMP decomposition" begin
        net = _make_isolated_load_network()
        d = [0.0, 0.4, 0.3]
        @test reference_buses(net) == [1, 3]

        state = DCPowerFlowState(net, [0.4, 0.0, 0.0], d)
        @test state.non_ref == [2]
        @test state.va[reference_buses(net)] == zeros(2)
        @test all(isfinite, state.f)

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)
        @test sol.psh[3] ≈ d[3] atol=1e-5
        @test sol.va[reference_buses(net)] ≈ zeros(2) atol=1e-8
        @test norm(kkt(flatten_variables(sol, prob), prob, d), Inf) < 1e-4

        congestion = calc_congestion_component(sol, net)
        energy = calc_energy_component(sol, net)
        @test calc_lmp(sol, net) ≈ energy + congestion atol=1e-10
        @test congestion[reference_buses(net)] == zeros(2)
    end

    @testset "fully isolated buses" begin
        net = _make_fully_isolated_network()
        d = [0.2, 0.3]
        @test reference_buses(net) == [1, 2]
        @test !isempty(sprint(show, MIME"text/plain"(), net))

        state = DCPowerFlowState(net, d, d)
        @test isempty(state.non_ref)
        @test isempty(state.f)
        @test state.va == zeros(2)
        @test Matrix(calc_sensitivity(state, :va, :d)) == zeros(2, 2)
        @test !isempty(sprint(show, MIME"text/plain"(), state))

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)
        @test sol.pg ≈ d atol=1e-5
        @test sol.va == zeros(2)
        @test norm(kkt(flatten_variables(sol, prob), prob, d), Inf) < 1e-4
        @test !isempty(sprint(show, MIME"text/plain"(), sol))
    end

    @testset "bridge opening resizes DC topology workspaces" begin
        net = _make_bridge_network()
        d = [0.0, 0.5, 0.0, 0.5]
        prob = DCOPFProblem(net, d)
        solve!(prob)

        @test reference_buses(net) == [1]
        dim_connected = kkt_dims(prob)
        out = zeros(net.n)
        jvp!(out, prob, :lmp, :d, ones(net.n))
        @test length(prob.cache.work) == dim_connected

        sw = copy(net.sw)
        sw[2] = 0.0
        update_switching!(prob, sw)
        @test reference_buses(net) == [1, 3]
        @test isnothing(prob.cache.work)
        @test isnothing(prob.cache.b_r_factor)
        @test length(prob.cons.ref) == 2

        sol = solve!(prob)
        @test kkt_dims(prob) == dim_connected + 1
        @test sol.va[reference_buses(net)] ≈ zeros(2) atol=1e-8
        @test norm(kkt(flatten_variables(sol, prob), prob, d), Inf) < 1e-4

        energy = calc_energy_component(sol, net)
        @test energy[1] ≈ energy[2] atol=1e-5
        @test energy[3] ≈ energy[4] atol=1e-5

        sensitivity = calc_sensitivity(prob, :pg, :d)
        @test all(isfinite, Matrix(sensitivity))
        jvp!(out, prob, :lmp, :d, ones(net.n))
        @test all(isfinite, out)
        @test length(prob.cache.work) == kkt_dims(prob)
        @test all(isfinite, vjp(prob, :lmp, :d, ones(net.n)))

        state = DCPowerFlowState(net, [0.5, 0.0, 0.5, 0.0], d)
        @test state.va[reference_buses(net)] == zeros(2)
        @test all(isfinite, Matrix(calc_sensitivity(state, :f, :d)))
    end

    @testset "reference bus cache follows direct topology mutation" begin
        net = _make_bridge_network()
        @test getfield(net, :topology_cache).initialized
        dim_connected = kkt_dims(net)
        @test reference_buses(net) == [1]

        # Returned references are defensive copies of the internal cache.
        refs = reference_buses(net)
        refs[1] = 4
        @test reference_buses(net) == [1]

        net.sw[2] = 0.0
        @test reference_buses(net) == [1, 3]
        @test kkt_dims(net) == dim_connected + 1

        net.sw[2] = 1.0
        @test reference_buses(net) == [1]
        @test kkt_dims(net) == dim_connected

        net.b[2] = 0.0
        @test reference_buses(net) == [1, 3]
        @test kkt_dims(net) == dim_connected + 1
    end

    @testset "bundled PowerModels disconnected cases" begin
        for case_name in ["case6.m"]
            @testset "$case_name" begin
                raw = load_raw_case(case_name)
                parsed = load_test_case(case_name)
                if isnothing(raw) || isnothing(parsed)
                    @test_skip false
                    continue
                end

                pm_result = PowerModels.solve_dc_opf(raw,
                    optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
                @test pm_result["termination_status"] in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)

                net = DCNetwork(parsed)
                d = calc_demand_vector(parsed)
                @test getfield(net, :topology_cache).initialized
                @test length(reference_buses(net)) > 1
                sol = solve!(DCOPFProblem(net, d))
                @test all(isfinite, sol.va)
                @test all(isfinite, sol.pg)
                @test sol.objective ≈ pm_result["objective"] rtol=1e-4 atol=1e-2
            end
        end
    end

    @testset "specialized PowerModels components stay unsupported" begin
        for case_name in ["case5_db.m", "case7_tplgy.m"]
            @testset "$case_name" begin
                raw = load_raw_case(case_name)
                parsed = load_test_case(case_name)
                if isnothing(raw) || isnothing(parsed)
                    @test_skip false
                    continue
                end

                specialized = sum(length(get(raw, key, Dict())) for key in ("dcline", "storage", "switch"))
                @test specialized > 0
                @test_throws ArgumentError DCNetwork(parsed)
            end
        end
    end
end
