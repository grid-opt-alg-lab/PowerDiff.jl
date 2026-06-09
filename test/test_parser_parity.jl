import PowerIO

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

    @testset "Parser contract" begin
        @test PowerDiff.parse_file(IOBuffer(_INLINE_CASE)) isa ParsedCase
        @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(_INLINE_CASE); backend=:native)
    end
end

# Field-for-field equality of two ParsedCase values; floats compared with ≈, ints with ==.
function _assert_parsedcase_equal(a::ParsedCase, b::ParsedCase, label)
    @testset "$label" begin
        @test a.baseMVA ≈ b.baseMVA
        @test length(a.bus) == length(b.bus)
        @test length(a.gen) == length(b.gen)
        @test length(a.branch) == length(b.branch)
        @test length(a.load) == length(b.load)
        @test length(a.shunt) == length(b.shunt)
        for (x, y) in zip(a.bus, b.bus)
            @test x.bus_i == y.bus_i
            @test x.bus_type == y.bus_type
            @test x.area == y.area && x.zone == y.zone
            @test x.vm ≈ y.vm && x.va ≈ y.va && x.base_kv ≈ y.base_kv
            @test x.vmax ≈ y.vmax && x.vmin ≈ y.vmin
        end
        for (x, y) in zip(a.gen, b.gen)
            @test x.gen_bus == y.gen_bus && x.gen_status == y.gen_status
            @test x.pg ≈ y.pg && x.qg ≈ y.qg && x.vg ≈ y.vg && x.mbase ≈ y.mbase
            @test x.pmax ≈ y.pmax && x.pmin ≈ y.pmin && x.qmax ≈ y.qmax && x.qmin ≈ y.qmin
            @test all(x.cost .≈ y.cost)
        end
        for (x, y) in zip(a.branch, b.branch)
            @test x.f_bus == y.f_bus && x.t_bus == y.t_bus && x.br_status == y.br_status
            @test x.br_r ≈ y.br_r && x.br_x ≈ y.br_x && x.br_b ≈ y.br_b
            @test x.rate_a ≈ y.rate_a && x.rate_b ≈ y.rate_b && x.rate_c ≈ y.rate_c
            @test x.tap ≈ y.tap && x.shift ≈ y.shift && x.angmin ≈ y.angmin && x.angmax ≈ y.angmax
        end
        for (x, y) in zip(a.load, b.load)
            @test x.load_bus == y.load_bus && x.status == y.status
            @test x.pd ≈ y.pd && x.qd ≈ y.qd
        end
        for (x, y) in zip(a.shunt, b.shunt)
            @test x.shunt_bus == y.shunt_bus && x.status == y.status
            @test x.gs ≈ y.gs && x.bs ≈ y.bs
        end
    end
end

@testset "PowerIO parser path and IO parity" begin
    # PowerIO is the only parser/data layer. Path parsing and IO parsing must
    # land on the same PowerDiff ParsedCase after normalization.
    if !PowerIO.library_available()
        @info "libpowerio_capi not found (set POWERIO_CAPI to a local build); skipping parser parity"
        @test_skip false
    else
        cases = filter(c -> isfile(joinpath(PD_PGLIB_DIR, c)),
                       ["pglib_opf_case5_pjm.m", "pglib_opf_case14_ieee.m", "pglib_opf_case30_ieee.m"])
        @test !isempty(cases)
        for c in cases
            path_case = PowerDiff.parse_file(c; library=:pglib)
            io_case = PowerDiff.parse_file(IOBuffer(read(joinpath(PD_PGLIB_DIR, c), String)))
            _assert_parsedcase_equal(path_case, io_case, c)
        end
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
