const _INLINE_CASE = """
function mpc = case_inline
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [1 2 50 10 1 -2 1 1.0 0 230 1 1.1 0.9; 2 1 0 0 0 0 1 1.0 0 230 1 1.1 0.9];
mpc.gen = [1 80 0 100 -100 1 100 1 150 0; 2 20 0 50 -50 1 100 0 50 0];
mpc.branch = [1 2 0.01 0.1 0.02 0 0 0 0 0 1 -360 360; 1 2 0.02 0.2 0.01 100 100 100 1 0 0 -30 30];
mpc.gencost = [2 0 0 3 0.01 2 3; 2 0 0 3 0.02 3 4];
mpc.areas = [1 1];
mpc.bus_name = ['one'; 'two'];
"""

@testset "MATPOWER Parser Semantics" begin
    @testset "Inline arrays and normalization" begin
        data = PowerDiff.parse_matpower(IOBuffer(_INLINE_CASE))

        @test data isa ParsedCase
        @test data.name == "case_inline"
        @test data.source_version == "2"
        @test data.baseMVA == 100.0
        @test length(data.bus) == 2
        @test length(data.gen) == 1
        @test length(data.branch) == 1
        @test length(data.load) == 1
        @test length(data.shunt) == 1
        @test data.bus[1].bus_type == 3
        @test data.bus[1].pd == 0.0
        @test data.load[1].pd == 0.5
        @test data.shunt[1].gs == 0.01
        @test data.branch[1].tap == 1.0
        @test data.branch[1].rate_a > 0
        @test data.branch[1].angmin ≈ -π / 3
        @test data.branch[1].angmax ≈ π / 3
        @test data.gen[1].cost == (100.0, 200.0, 3.0)
    end

    @testset "Multiline arrays and artifact path" begin
        parsed = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)
        @test parsed isa ParsedCase
        @test length(parsed.bus) == 14
        @test length(parsed.branch) == 20
        @test PowerDiff.get_path(:pglib) == PD_PGLIB_DIR
    end

    @testset "Rejected inputs" begin
        @test_throws ArgumentError PowerDiff.parse_file("case.raw")
        @test_throws ArgumentError PowerDiff.parse_file("case.json")
        @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(_INLINE_CASE); filetype="json")
        @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(_INLINE_CASE); unsupported=true)
        @test_throws ArgumentError PowerDiff.get_path(:unknown)

        unsupported = replace(_INLINE_CASE, "mpc.areas = [1 1];" => "mpc.storage = [1 1];")
        @test_throws ArgumentError PowerDiff.parse_matpower(IOBuffer(unsupported))

        invalid = replace(_INLINE_CASE, "0.01 0.1" => "NaN 0.1")
        @test_throws ArgumentError PowerDiff.parse_matpower(IOBuffer(invalid))

        pwl = replace(_INLINE_CASE, "2 0 0 3 0.01 2 3" => "1 0 0 3 0.01 2 3")
        @test_throws ArgumentError PowerDiff.parse_matpower(IOBuffer(pwl))

        quartic = replace(_INLINE_CASE, "2 0 0 3 0.01 2 3" => "2 0 0 4 1 0.01 2 3")
        @test_throws ArgumentError PowerDiff.parse_matpower(IOBuffer(quartic))
    end
end

@testset "Typed AC Pi Model" begin
    buses = [
        ParsedBus(1, 3, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9),
        ParsedBus(2, 1, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9),
        ParsedBus(3, 1, 0.0, 0.0, 0.0, 0.0, 1, 1.0, 0.0, 230.0, 1, 1.1, 0.9),
    ]
    gens = [
        ParsedGen(1, 1, 0.5, 0.0, 1.0, -1.0, 1.0, 100.0, 1, 2.0, 0.0, (1.0, 1.0, 0.0)),
    ]
    branches = [
        ParsedBranch(1, 1, 2, 0.01, 0.10, 0.02, 2.0, 2.0, 2.0, 1.05, 0.12, 1, -π / 3, π / 3),
        ParsedBranch(2, 1, 2, 0.02, 0.20, 0.01, 2.0, 2.0, 2.0, 1.00, 0.00, 1, -π / 3, π / 3),
        ParsedBranch(3, 2, 3, 0.01, 0.15, 0.03, 2.0, 2.0, 2.0, 0.97, -0.08, 1, -π / 3, π / 3),
    ]
    data = ParsedCase("pi_model", "2", 100.0, buses, gens, branches, ParsedLoad[], ParsedShunt[])
    net = ACNetwork(data)
    v = [1.01 + 0.02im, 0.98 - 0.04im, 1.02 + 0.01im]

    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    expected_current = ComplexF64[]
    for l in 1:net.m
        y = net.g[l] + im * net.b[l]
        tap = net.tap[l] * cis(net.shift[l])
        yff = (y + net.g_fr[l] + im * net.b_fr[l]) / abs2(tap)
        yft = -y / conj(tap)
        ytf = -y / tap
        ytt = y + net.g_to[l] + im * net.b_to[l]
        append!(rows, (net.f_bus[l], net.f_bus[l], net.t_bus[l], net.t_bus[l]))
        append!(cols, (net.f_bus[l], net.t_bus[l], net.f_bus[l], net.t_bus[l]))
        append!(vals, (yff, yft, ytf, ytt))
        push!(expected_current, yff * v[net.f_bus[l]] + yft * v[net.t_bus[l]])
    end
    expected_y = sparse(rows, cols, vals, net.n, net.n)

    @test Matrix(admittance_matrix(net)) ≈ Matrix(expected_y)
    @test branch_current(net, v) ≈ expected_current
    @test branch_power(net, v) ≈ v[net.f_bus] .* conj.(expected_current)
end

@testset "ParsedCase Status Filtering" begin
    parsed = PowerDiff.parse_matpower(IOBuffer(_INLINE_CASE))
    @test length(parsed.gen) == 1
    @test length(parsed.branch) == 1
end
