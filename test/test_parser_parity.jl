@testset "MATPOWER Parser Semantics" begin
    local pm_cases = [
        "case14.m",
        "case3.m",
        "case5_dc.m",
        "case5_strg.m",
        "case5_sw.m",
        "case7_tplgy.m",
        "case3_tnep.m",
    ]

    function compare_parsed_case(case_path, parsed::PowerDiff.ParsedCase)
        upstream = PowerModels.parse_file(case_path)
        up_buses = upstream["bus"]
        up_gens = upstream["gen"]
        up_branches = upstream["branch"]
        up_loads = get(upstream, "load", Dict{String,Any}())
        up_shunts = get(upstream, "shunt", Dict{String,Any}())

        expected_cost(cost) =
            length(cost) >= 3 ? (cost[end-2], cost[end-1], cost[end]) :
            length(cost) == 2 ? (0.0, cost[1], cost[2]) :
            length(cost) == 1 ? (0.0, 0.0, cost[1]) :
            (0.0, 0.0, 0.0)

        @test parsed.baseMVA ≈ upstream["baseMVA"]
        @test parsed.source_version == upstream["source_version"]
        @test length(parsed.bus) == length(up_buses)
        @test length(parsed.gen) == length(up_gens)
        @test length(parsed.branch) == length(up_branches)
        @test length(parsed.load) == length(up_loads)
        @test length(parsed.shunt) == length(up_shunts)

        for bus in parsed.bus
            up = up_buses[string(bus.bus_i)]
            @test bus.bus_type == up["bus_type"]
            @test bus.pd ≈ get(up, "pd", 0.0)
            @test bus.qd ≈ get(up, "qd", 0.0)
            @test bus.gs ≈ get(up, "gs", 0.0)
            @test bus.bs ≈ get(up, "bs", 0.0)
            @test bus.vm ≈ get(up, "vm", 1.0)
            @test bus.va ≈ get(up, "va", 0.0)
        end

        for gen in parsed.gen
            up = up_gens[string(gen.index)]
            @test gen.gen_bus == up["gen_bus"]
            @test gen.pg ≈ get(up, "pg", 0.0)
            @test gen.qg ≈ get(up, "qg", 0.0)
            @test gen.pmax ≈ up["pmax"]
            @test gen.pmin ≈ up["pmin"]
            @test gen.cost == expected_cost(up["cost"])
        end

        for branch in parsed.branch
            up = up_branches[string(branch.index)]
            @test branch.f_bus == up["f_bus"]
            @test branch.t_bus == up["t_bus"]
            @test branch.br_r ≈ up["br_r"]
            @test branch.br_x ≈ up["br_x"]
            @test branch.br_b ≈ (get(up, "b_fr", 0.0) + get(up, "b_to", 0.0))
            @test branch.rate_a ≈ get(up, "rate_a", 0.0)
            @test branch.br_status == get(up, "br_status", 1)
            @test branch.angmin ≈ get(up, "angmin", -pi) atol = 1e-5
            @test branch.angmax ≈ get(up, "angmax", pi) atol = 1e-5
        end

        for load in parsed.load
            up = up_loads[string(load.index)]
            @test load.load_bus == up["load_bus"]
            @test load.pd ≈ get(up, "pd", 0.0)
            @test load.qd ≈ get(up, "qd", 0.0)
            @test load.status == get(up, "status", 1)
        end

        for shunt in parsed.shunt
            up = up_shunts[string(shunt.index)]
            @test shunt.shunt_bus == up["shunt_bus"]
            @test shunt.gs ≈ get(up, "gs", 0.0)
            @test shunt.bs ≈ get(up, "bs", 0.0)
            @test shunt.status == get(up, "status", 1)
        end
    end

    for case_name in pm_cases
        case_path = joinpath(PM_DATA_DIR, case_name)
        @testset "$case_name" begin
            parsed = PowerDiff.parse_file(case_path)
            @test parsed isa PowerDiff.ParsedCase
            compare_parsed_case(case_path, parsed)
        end
    end

    local pglib_cases = [
        "pglib_opf_case3_lmbd.m",
        "pglib_opf_case5_pjm.m",
        "pglib_opf_case14_ieee.m",
    ]

    for case_name in pglib_cases
        case_path = joinpath(PD_PGLIB_DIR, case_name)
        @testset "pglib/$case_name" begin
            parsed = PowerDiff.parse_file(case_name; library=:pglib)
            @test parsed isa PowerDiff.ParsedCase
            compare_parsed_case(case_path, parsed)
        end
    end
end


@testset "ParsedCase Constructor Parity" begin
    @testset "AC constructor matches dict path" begin
        case_path = joinpath(PM_DATA_DIR, "case14.m")
        raw = PowerModels.parse_file(case_path)
        parsed = PowerDiff.parse_file(case_path)

        net_raw = ACNetwork(raw)
        net_parsed = ACNetwork(parsed)
        @test norm(Matrix(admittance_matrix(net_raw) - admittance_matrix(net_parsed)), Inf) ≤ 1e-12

        sol_raw = solve!(ACOPFProblem(raw; silent=true))
        sol_parsed = solve!(ACOPFProblem(parsed; silent=true))
        @test sol_parsed.objective ≈ sol_raw.objective atol = 1e-9 rtol = 1e-9
    end

    @testset "DC constructor matches dict path" begin
        case_path = joinpath(PM_DATA_DIR, "case5.m")
        raw = PowerModels.parse_file(case_path)
        parsed = PowerDiff.parse_file(case_path)

        net_raw = DCNetwork(raw)
        net_parsed = DCNetwork(parsed)
        @test norm(Matrix(net_raw.A - net_parsed.A), Inf) ≤ 1e-12
        @test norm(net_raw.b - net_parsed.b, Inf) ≤ 1e-12

        sol_raw = solve!(DCOPFProblem(raw))
        sol_parsed = solve!(DCOPFProblem(parsed))
        @test sol_parsed.objective ≈ sol_raw.objective atol = 1e-9 rtol = 1e-9
    end

    @testset "Inactive elements stay inactive" begin
        data = PowerDiff.ParsedCase(
            "status_case", "2", 1.0,
            [
                PowerDiff.ParsedBus(1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 1.0, 1, 1.1, 0.9),
                PowerDiff.ParsedBus(2, 2, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 1.0, 1, 1.1, 0.9),
            ],
            [
                PowerDiff.ParsedGen(1, 1, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 1, 2.0, 0.0, (0.0, 1.0, 0.0)),
                PowerDiff.ParsedGen(2, 2, 1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0, 2.0, 0.0, (0.0, 2.0, 0.0)),
            ],
            [
                PowerDiff.ParsedBranch(1, 1, 2, 0.0, 0.1, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1, -π, π),
                PowerDiff.ParsedBranch(2, 1, 2, 0.0, 0.1, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0, -π, π),
            ],
            PowerDiff.ParsedLoad[],
            PowerDiff.ParsedShunt[],
        )

        dc = DCNetwork(data)
        ac = ACNetwork(data)
        @test dc.m == 1
        @test dc.k == 1
        @test ac.m == 1
        @test length(ac.gen_bus) == 1
    end
end
