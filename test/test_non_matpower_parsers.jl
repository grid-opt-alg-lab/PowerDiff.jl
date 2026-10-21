using PowerIO

function _assert_network_tables_compatible(actual, baseline; label, demand_rtol=1e-8)
    @testset "$label tables" begin
        @test length(actual.bus) == length(baseline.bus)
        @test length(actual.gen) == length(baseline.gen)
        @test length(actual.branch) == length(baseline.branch)
        @test sum(calc_demand_vector(actual)) ≈ sum(calc_demand_vector(baseline)) rtol=demand_rtol atol=1e-8
        @test all(isfinite, [br.rate_a for br in actual.branch])
        @test all(>(0), [br.rate_a for br in actual.branch])
        @test sum(br.b_fr + br.b_to for br in actual.branch) ≈
              sum(br.b_fr + br.b_to for br in baseline.branch) rtol=1e-6 atol=1e-8
    end
end

function _assert_constructs_and_solves(data; label)
    @testset "$label constructors" begin
        dc = DCNetwork(data)
        @test dc.n == length(data.bus)
        @test dc.k == length(data.gen)
        @test dc.m == length(data.branch)
        @test all(isfinite, dc.b)

        dc_prob = DCOPFProblem(dc)
        dc_sol = solve!(dc_prob)
        @test all(isfinite, dc_sol.va)
        @test all(isfinite, dc_sol.pg)
        @test all(isfinite, dc_sol.f)

        ac = ACNetwork(data)
        @test size(admittance_matrix(ac)) == (ac.n, ac.n)
        ac_prob = ACOPFProblem(ac; silent=true)
        @test ac_prob.n_gen == length(data.gen)
    end
end

@testset "Non MATPOWER Parser Support" begin
    matpower_net = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)
    baseline = PowerDiff._network_data(matpower_net)

    @testset "PowerModels JSON" begin
        # `emit` returns an `EmitResult`: the artifacts it wrote, the fidelity of the
        # conversion, and the writer's findings as `Diagnostic` records.
        result = PowerIO.emit(matpower_net, "powermodels-json")
        @test result.layout == "file"
        @test isempty(result.diagnostics)
        text = result.text
        @test text isa String

        parsed = PowerDiff.parse_file(IOBuffer(text); from=:powermodels)
        @test parsed.sources[1].format == "powermodels-json"
        data = PowerDiff._network_data(parsed)
        _assert_network_tables_compatible(data, baseline; label="PowerModels JSON")
        _assert_constructs_and_solves(data; label="PowerModels JSON")

        mktempdir() do dir
            path = joinpath(dir, "case14.json")
            written = PowerIO.emit(matpower_net, "powermodels-json", path)
            @test written.layout == "file"
            @test isfile(written.artifacts[1].path)
            parsed_path = PowerDiff.parse_file(path; from="powermodels-json")
            @test parsed_path.sources[1].format == "powermodels-json"
            _assert_network_tables_compatible(
                PowerDiff._network_data(parsed_path), baseline; label="PowerModels JSON path")
        end
    end

    @testset "Egret JSON" begin
        result = PowerIO.emit(matpower_net, "egret-json")
        @test isempty(result.diagnostics)
        text = result.text

        parsed = PowerDiff.parse_file(IOBuffer(text); from=:egret)
        @test parsed.sources[1].format == "egret-json"
        data = PowerDiff._network_data(parsed)
        _assert_network_tables_compatible(data, baseline; label="Egret JSON")
        _assert_constructs_and_solves(data; label="Egret JSON")

        @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(text); filetype="json")
    end

    @testset "PSS/E RAW" begin
        result = PowerIO.emit(matpower_net, "psse")
        # A `Diagnostic` is a record, not a line of text: branch on `code` and
        # `severity`, never on the message, which carries no stability promise.
        @test result.diagnostics isa Vector{PowerIO.Diagnostic}
        @test all(d -> d.severity in (:error, :warning, :remark, :note), result.diagnostics)
        @test !any(d -> d.severity === :error, result.diagnostics)
        text = result.text

        parsed = PowerDiff.parse_file(IOBuffer(text); from=:psse)
        @test parsed.sources[1].format == "psse"
        data = PowerDiff._network_data(parsed)
        _assert_network_tables_compatible(data, baseline; label="PSS/E RAW", demand_rtol=1e-5)
        _assert_constructs_and_solves(data; label="PSS/E RAW")

        mktempdir() do dir
            path = joinpath(dir, "case14.raw")
            written = PowerIO.emit(matpower_net, "psse", path)
            @test isfile(written.artifacts[1].path)
            parsed_path = PowerDiff.parse_file(path)
            @test parsed_path.sources[1].format == "psse"
            _assert_network_tables_compatible(
                PowerDiff._network_data(parsed_path), baseline; label="PSS/E RAW path", demand_rtol=1e-5)
        end
    end
end

@testset "Programmatic AC OPF constructor validation" begin
    buses = [pd_bus(10, 1), pd_bus(20, 1)]
    gens = [pd_gen(1, 10; pg=0.0, qmax=1.0, qmin=-1.0, pmax=2.0, pmin=0.0, cost=(1.0, 1.0, 0.0))]
    branches = [pd_branch(1, 10, 20; br_r=0.01, br_x=0.1, rate_a=2.0)]
    data = pd_case(buses, gens, branches; name="no_ref_bus")

    net = ACNetwork(data; idx_slack=2)
    @test net.idx_slack == 2
    @test net.ref_bus_keys == [2]
    prob = ACOPFProblem(net; silent=true)
    @test prob.data.ref_bus_keys == [2]

    typed_ref_data = pd_case(
        [pd_bus(10, 3), pd_bus(20, 1)],
        gens,
        branches;
        name="typed_ref_bus",
    )
    override_net = ACNetwork(typed_ref_data; idx_slack=2)
    @test override_net.idx_slack == 2
    @test override_net.ref_bus_keys == [2]
    @test ACOPFProblem(override_net; silent=true).data.ref_bus_keys == [2]
    @test_throws ArgumentError ACNetwork(typed_ref_data; idx_slack=0)
    @test_throws ArgumentError ACNetwork(typed_ref_data; idx_slack=3)

    y = ComplexF64[1 -1; -1 1]
    raw_net = ACNetwork(y)
    @test_throws ArgumentError ACOPFProblem(raw_net; silent=true)
end
