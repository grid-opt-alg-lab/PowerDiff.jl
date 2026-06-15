import PowerIO

const _INLINE_CASE = """
function mpc = case_inline
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [1 2 50 10 1 -2 1 1.0 0 230 1 1.1 0.9; 2 1 0 0 0 0 1 1.0 0 230 1 1.1 0.9];
mpc.gen = [1 80 0 100 -100 1 100 1 150 0; 2 20 0 50 -50 1 100 0 50 0];
mpc.branch = [1 2 0.01 0.1 0.02 0 0 0 0 0 0 -360 360; 1 2 0.02 0.2 0.01 100 100 100 1 0 1 -60 60];
mpc.gencost = [2 0 0 3 0.01 2 3; 2 0 0 3 0.02 3 4];
mpc.areas = [1 1];
mpc.bus_name = ['one'; 'two'];
"""

# `parse_file`/`parse_matpower` return a PowerIO.Network; `_network_data` applies
# PowerDiff's normalization (per-unit via PowerIO, cost right-align, rate_a fallback,
# angle defaults, storage/HVDC rejection) into the tables the constructors consume.
_inline_data() = PowerDiff._network_data(PowerDiff.parse_matpower(IOBuffer(_INLINE_CASE)))

@testset "MATPOWER Parser Semantics" begin
    @testset "Inline arrays and normalization" begin
        data = _inline_data()

        @test data isa NamedTuple
        @test data.name == "case_inline"
        @test data.baseMVA == 100.0
        @test length(data.bus) == 2
        @test length(data.gen) == 1
        @test length(data.branch) == 1
        @test data.gen[1].index == 1
        @test data.branch[1].index == 2
        @test data.bus[1].bus_type == 3
        # Loads and shunts are aggregated into per-bus values.
        @test data.bus[1].pd == 0.5
        @test data.bus[1].gs == 0.01
        # Shunts are also re-exposed as a per-bus table (bus 1: Gs=1, Bs=-2 -> 0.01, -0.02 pu).
        @test length(data.shunt) == 1
        @test data.shunt[1].shunt_bus == 1
        @test data.shunt[1].gs ≈ 0.01 && data.shunt[1].bs ≈ -0.02
        @test data.branch[1].tap == 1.0
        @test data.branch[1].rate_a > 0
        @test data.branch[1].angmin ≈ -π / 3
        @test data.branch[1].angmax ≈ π / 3
        @test data.gen[1].cost == (100.0, 200.0, 3.0)
    end

    @testset "Multiline arrays and artifact path" begin
        parsed = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)
        @test parsed isa PowerIO.Network
        nd = PowerDiff._network_data(parsed)
        @test length(nd.bus) == 14
        @test length(nd.branch) == 20
        @test PowerDiff.get_path(:pglib) == PD_PGLIB_DIR
    end

    @testset "Rejected inputs" begin
        @test_throws ArgumentError PowerDiff.parse_file("case.json")
        @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(_INLINE_CASE); filetype="json")
        @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(_INLINE_CASE); unsupported=true)
        @test_throws ArgumentError PowerDiff.get_path(:unknown)

        # Modeling-level rejections happen when the network tables are built.
        unsupported = replace(_INLINE_CASE, "mpc.areas = [1 1];" => "mpc.storage = [1 1];")
        @test_throws ArgumentError PowerDiff._network_data(PowerDiff.parse_matpower(IOBuffer(unsupported)))

        invalid = replace(_INLINE_CASE, "0.01 0.1" => "NaN 0.1")
        @test_throws ArgumentError PowerDiff._network_data(PowerDiff.parse_matpower(IOBuffer(invalid)))

        pwl = replace(_INLINE_CASE, "2 0 0 3 0.01 2 3" => "1 0 0 3 0.01 2 3")
        @test_throws ArgumentError PowerDiff._network_data(PowerDiff.parse_matpower(IOBuffer(pwl)))

        quartic = replace(_INLINE_CASE, "2 0 0 3 0.01 2 3" => "2 0 0 4 1 0.01 2 3")
        @test_throws ArgumentError PowerDiff._network_data(PowerDiff.parse_matpower(IOBuffer(quartic)))
    end

    @testset "Parser contract" begin
        @test PowerDiff.parse_file(IOBuffer(_INLINE_CASE)) isa PowerIO.Network
        @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(_INLINE_CASE); backend=:native)
    end
end

# Field-for-field equality of two PowerDiff network tables; floats with ≈, ints with ==.
function _assert_netdata_equal(a, b, label)
    @testset "$label" begin
        @test a.baseMVA ≈ b.baseMVA
        @test length(a.bus) == length(b.bus)
        @test length(a.gen) == length(b.gen)
        @test length(a.branch) == length(b.branch)
        for (x, y) in zip(a.bus, b.bus)
            @test x.bus_i == y.bus_i
            @test x.bus_type == y.bus_type
            @test x.pd ≈ y.pd && x.qd ≈ y.qd && x.gs ≈ y.gs && x.bs ≈ y.bs
            @test x.vm ≈ y.vm && x.va ≈ y.va && x.vmin ≈ y.vmin && x.vmax ≈ y.vmax
        end
        for (x, y) in zip(a.gen, b.gen)
            @test x.gen_bus == y.gen_bus
            @test x.pg ≈ y.pg && x.qg ≈ y.qg
            @test x.pmin ≈ y.pmin && x.pmax ≈ y.pmax && x.qmin ≈ y.qmin && x.qmax ≈ y.qmax
            @test all(x.cost .≈ y.cost)
        end
        for (x, y) in zip(a.branch, b.branch)
            @test x.f_bus == y.f_bus && x.t_bus == y.t_bus
            @test x.br_r ≈ y.br_r && x.br_x ≈ y.br_x && x.br_b ≈ y.br_b
            @test x.rate_a ≈ y.rate_a
            @test x.tap ≈ y.tap && x.shift ≈ y.shift && x.angmin ≈ y.angmin && x.angmax ≈ y.angmax
        end
    end
end

@testset "PowerIO parser path and IO parity" begin
    # PowerIO is the only parser/data layer. Path parsing and IO parsing must land on
    # the same PowerDiff network tables after normalization.
    if !PowerIO.library_available()
        @info "libpowerio_capi not found (set POWERIO_CAPI to a local build); skipping parser parity"
        @test_skip false
    else
        cases = filter(c -> isfile(joinpath(PD_PGLIB_DIR, c)),
                       ["pglib_opf_case5_pjm.m", "pglib_opf_case14_ieee.m", "pglib_opf_case30_ieee.m"])
        @test !isempty(cases)
        for c in cases
            path_data = PowerDiff._network_data(PowerDiff.parse_file(c; library=:pglib))
            io_data = PowerDiff._network_data(
                PowerDiff.parse_file(IOBuffer(read(joinpath(PD_PGLIB_DIR, c), String))))
            _assert_netdata_equal(path_data, io_data, c)
        end
    end
end

@testset "Typed AC Pi Model" begin
    buses = [
        pd_bus(1, 3; vmax=1.1, vmin=0.9),
        pd_bus(2, 1; vmax=1.1, vmin=0.9),
        pd_bus(3, 1; vmax=1.1, vmin=0.9),
    ]
    gens = [
        pd_gen(1, 1; pg=0.5, qmax=1.0, qmin=-1.0, vg=1.0, pmax=2.0, pmin=0.0, cost=(1.0, 1.0, 0.0)),
    ]
    branches = [
        pd_branch(1, 1, 2; br_r=0.01, br_x=0.10, br_b=0.02, rate_a=2.0, rate_b=2.0, rate_c=2.0, tap=1.05, shift=0.12, angmin=-π / 3, angmax=π / 3),
        pd_branch(2, 1, 2; br_r=0.02, br_x=0.20, br_b=0.01, rate_a=2.0, rate_b=2.0, rate_c=2.0, tap=1.00, shift=0.00, angmin=-π / 3, angmax=π / 3),
        pd_branch(3, 2, 3; br_r=0.01, br_x=0.15, br_b=0.03, rate_a=2.0, rate_b=2.0, rate_c=2.0, tap=0.97, shift=-0.08, angmin=-π / 3, angmax=π / 3),
    ]
    data = pd_case(buses, gens, branches; name="pi_model")
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

@testset "Status Filtering" begin
    parsed = _inline_data()
    @test length(parsed.gen) == 1
    @test length(parsed.branch) == 1
end
