# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The PowerDiff/PowerIO seam: format routing, the one-pass ingest and its findings,
# and the bounds a case may leave unstated.

import JuMP
import PowerIO

@testset "PowerIO format routing" begin
    # PowerIO owns the format vocabulary. PowerDiff normalizes its own historical
    # short spellings and hands everything else straight over, so a reader PowerIO
    # ships is reachable without a PowerDiff release.
    @test PowerDiff._format_token(:m) == "matpower"
    @test PowerDiff._format_token(".m") == "matpower"
    @test PowerDiff._format_token(:raw) == "psse"
    @test PowerDiff._format_token(:aux) == "powerworld"
    @test PowerDiff._format_token(:pm) == "powermodels-json"
    @test PowerDiff._format_token(:powermodels) == "powermodels-json"
    @test PowerDiff._format_token(:egret) == "egret-json"

    # Tokens PowerDiff has never named: passed through for PowerIO to resolve, not
    # refused here. This is the property that keeps the two in step, and it covers
    # the distribution readers too -- what refuses those is the value they parse to.
    for token in (:psse34, :psse35, :pandapower, :pypsa, :pslf, :epc, :pwb,
                  :gridfm, :goc3, :surge, :opfdata, :xiidm, :cgmes, :dss, :pmd,
                  Symbol("pypsa-csv"))
        @test PowerDiff._format_token(token) == lowercase(String(token))
    end

    # Two hints must agree, and the alias layer is what lets them.
    @test PowerDiff._powerio_format_hint(:matpower, ".m") == "matpower"
    @test_throws ArgumentError PowerDiff._powerio_format_hint(:matpower, :psse)

    # A bare `json` names a container, not a reader.
    @test_throws ArgumentError PowerDiff._format_token(:json)
    @test_throws ArgumentError PowerDiff._format_token(".json")

    @test_throws ArgumentError PowerDiff._format_token("")
end

@testset "Only a balanced transmission network is modeled" begin
    # PowerIO returns one of twenty value kinds, and only a balanced transmission
    # network has what PowerDiff differentiates. The refusal is the type of what was
    # parsed, so one rule covers distribution cases, series carriers and calculation
    # instances alike, with no token list here to trail what PowerIO reads.
    net = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)
    @test PowerDiff._require_balanced(net) === net

    dss = """
    New Circuit.mini basekv=12.47 pu=1.0 phases=3 bus1=sourcebus
    New Linecode.lc1 nphases=3 r1=0.1 x1=0.2 c1=0 units=km
    New Line.l1 bus1=sourcebus bus2=b1 linecode=lc1 length=1 units=km
    New Load.ld1 bus1=b1 phases=3 kV=12.47 kW=100 kvar=30
    Set voltagebases=[12.47]
    Calcvoltagebases
    Solve
    """
    distribution = PowerIO.parse(IOBuffer(dss); format="dss", name="mini.dss")
    @test distribution isa PowerIO.PioModule{PowerIO.MulticonductorNetwork}

    err = try
        PowerDiff._require_balanced(distribution)
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("balanced transmission networks", sprint(showerror, err))
    @test occursin("MulticonductorNetwork", sprint(showerror, err))

    @test_throws ArgumentError PowerDiff.parse_file(IOBuffer(dss); from=:dss)
end

@testset "PowerIO states the network, PowerDiff selects from it" begin
    net = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)
    tables = PowerDiff._network_data(net)

    # Both network types read the same tables, so they cannot describe different
    # cases, and both accept the module or the network inside it.
    dc = DCNetwork(net)
    ac = ACNetwork(net)
    @test dc.n == ac.n
    @test dc.m == ac.m
    @test dc.id_map.bus_ids == ac.id_map.bus_ids
    @test dc.id_map.branch_ids == ac.id_map.branch_ids
    @test DCNetwork(net.value).id_map.bus_ids == dc.id_map.bus_ids

    # `to_powerdata` is unfiltered: row `i` is the source row number and `status`
    # says whether the row is in service, so an out-of-service row leaves a gap
    # rather than renumbering the rows that remain.
    raw = PowerIO.to_powerdata(net)
    @test [Int(g.i) for g in raw.gen if Int(g.status) != 0] == [g.index for g in tables.gen]
    @test [Int(b.i) for b in raw.branch if Int(b.status) != 0] == [br.index for br in tables.branch]

    # The series admittance is read off PowerIO's terminal coefficients rather than
    # derived, and the incidence matrix labels the same edges the tables name.
    for (l, br) in enumerate(tables.branch)
        @test br.g ≈ Float64(raw.branch[br.index].c7) - br.g_to
        @test br.b ≈ Float64(raw.branch[br.index].c8) - br.b_to
        @test dc.A[l, dc.id_map.bus_to_idx[br.f_bus]] == 1.0
        @test dc.A[l, dc.id_map.bus_to_idx[br.t_bus]] == -1.0
    end
    @test dc.b == [br.b for br in tables.branch]
    @test ac.g == [br.g for br in tables.branch]

    # Reader findings are records on the module, not log output and not a wrapper.
    @test net.diagnostics isa Vector{PowerIO.Diagnostic}
    @test all(d -> d.code isa String && !isempty(d.code), net.diagnostics)
    @test net.sources[1].format == "matpower"
    @test net.producer.name isa String
end

@testset "A source that states no thermal limit" begin
    # MATPOWER spells an absent rating `0`, and PowerIO carries that out of
    # `to_powerdata` as `Inf`. Both mean the same thing, both take the synthesized
    # limit, and neither may reach the solver as an unbounded flow.
    template = """
    function mpc = case_rate
    mpc.version = '2';
    mpc.baseMVA = 100;
    mpc.bus = [1 3 0 0 0 0 1 1.0 0 230 1 1.1 0.9; 2 1 50 10 0 0 1 1.0 0 230 1 1.1 0.9];
    mpc.gen = [1 60 0 100 -100 1 100 1 150 0];
    mpc.branch = [1 2 0.01 0.1 0.02 %RATE% %RATE% %RATE% 0 0 1 -60 60];
    mpc.gencost = [2 0 0 3 0.01 2 3];
    """
    absent = PowerDiff.parse_matpower(IOBuffer(replace(template, "%RATE%" => "0")))
    huge = PowerDiff.parse_matpower(IOBuffer(replace(template, "%RATE%" => "1e6")))

    nd = PowerDiff._network_data(absent)
    @test all(br -> isfinite(br.rate_a) && br.rate_a > 0, nd.branch)
    @test all(isfinite, DCNetwork(absent).fmax)

    a = solve!(DCOPFProblem(DCNetwork(absent)))
    h = solve!(DCOPFProblem(DCNetwork(huge)))
    @test a.pg ≈ h.pg atol=1e-6
    @test a.f ≈ h.f atol=1e-6
    @test a.psh ≈ h.psh atol=1e-6
end

@testset "Isolated buses do not reach the model" begin
    # `to_powerdata` no longer drops a `type == 4` bus, so PowerDiff drops it and
    # every branch and generator standing on it. Bus 3 sits in the middle of the bus
    # table, so a missed re-index would move branch 2 onto the wrong bus.
    isolated = """
    function mpc = case_isolated
    mpc.version = '2';
    mpc.baseMVA = 100;
    mpc.bus = [1 3 0 0 0 0 1 1.0 0 230 1 1.1 0.9; 2 1 30 5 0 0 1 1.0 0 230 1 1.1 0.9; 3 4 25 5 0 0 1 1.0 0 230 1 1.1 0.9; 4 1 20 5 0 0 1 1.0 0 230 1 1.1 0.9];
    mpc.gen = [1 60 0 100 -100 1 100 1 150 0];
    mpc.branch = [1 2 0.01 0.1 0.02 100 100 100 0 0 1 -60 60; 2 4 0.02 0.2 0.01 100 100 100 0 0 1 -60 60; 2 3 0.03 0.3 0.01 100 100 100 0 0 1 -60 60];
    mpc.gencost = [2 0 0 3 0.01 2 3];
    """
    nd = PowerDiff._network_data(PowerDiff.parse_matpower(IOBuffer(isolated)))
    @test [b.bus_i for b in nd.bus] == [1, 2, 4]
    @test all(b -> b.bus_type != 4, nd.bus)
    # The branch to the isolated bus goes with it; the survivors keep their source
    # row numbers and land on the right buses.
    @test [br.index for br in nd.branch] == [1, 2]
    @test (nd.branch[2].f_bus, nd.branch[2].t_bus) == (2, 4)
    # Demand at an isolated bus leaves the model with the bus.
    @test sum(b.pd for b in nd.bus) ≈ 0.50
end

@testset "Terminal charging is carried, not averaged" begin
    # Each terminal states its own charging admittance. Summing the two and splitting
    # the total evenly, as PowerDiff once did, loses an asymmetric source's fidelity.
    net = ACNetwork(pd_case([pd_bus(1, 3), pd_bus(2, 1; pd=0.2)],
                            [pd_gen(1, 1; pmax=2.0, cost=(1.0, 1.0, 0.0))],
                            [pd_branch(1, 1, 2; br_r=0.01, br_x=0.1, rate_a=2.0,
                                       g_fr=0.001, b_fr=0.004, g_to=0.002, b_to=0.012)]))
    @test net.b_fr == [0.004] && net.b_to == [0.012]
    @test net.g_fr == [0.001] && net.g_to == [0.002]
    Y = Matrix(admittance_matrix(net))
    y = inv(complex(0.01, 0.1))
    @test Y[1, 1] ≈ y + complex(0.001, 0.004)
    @test Y[2, 2] ≈ y + complex(0.002, 0.012)
end

@testset "Absent reactive limits" begin
    # PowerIO 0.9 carries a bound the case does not state as ±Inf rather than refusing
    # the case (stock case9241pegase leaves the reactive limits off seven generators).
    # PowerDiff leaves the bound off the solver model and pins its multiplier to zero,
    # which is what a solver reports for a bound it was never given.
    buses = [pd_bus(1, 3; vmax=1.1, vmin=0.9), pd_bus(2, 1; pd=0.3, qd=0.1, vmax=1.1, vmin=0.9)]
    gens = [pd_gen(1, 1; pg=0.3, qmin=-Inf, qmax=Inf, pmax=2.0, pmin=0.0, cost=(0.0, 1.0, 0.0))]
    branches = [pd_branch(1, 1, 2; br_r=0.01, br_x=0.1, rate_a=2.0)]
    data = pd_case(buses, gens, branches; name="absent_q_limits")

    net = ACNetwork(data)
    @test net.qmin == [-Inf]
    @test net.qmax == [Inf]
    @test PowerDiff._absent_bound(net.qmin[1])
    @test PowerDiff._absent_bound(net.qmax[1])
    @test !PowerDiff._absent_bound(net.pmax[1])

    prob = ACOPFProblem(net; silent=true)
    @test !JuMP.has_lower_bound(prob.qg[1])
    @test !JuMP.has_upper_bound(prob.qg[1])
    @test JuMP.has_lower_bound(prob.pg[1])

    sol = solve!(prob)
    @test all(isfinite, sol.qg)
    # An absent bound has no multiplier. Zero is the value, not a placeholder.
    @test sol.rho_qg_lb == [0.0]
    @test sol.rho_qg_ub == [0.0]

    # The KKT residual and Jacobian stay finite: the two rows read `ρ = 0`.
    idx = kkt_indices(prob)
    K = kkt(flatten_variables(sol, prob), prob)
    @test all(isfinite, K)
    @test K[idx.rho_qg_lb[1]] == sol.rho_qg_lb[1]
    @test K[idx.rho_qg_ub[1]] == sol.rho_qg_ub[1]

    J = calc_kkt_jacobian(prob; sol=sol)
    @test all(isfinite, nonzeros(J))
    @test J[idx.rho_qg_lb[1], idx.rho_qg_lb[1]] == 1.0
    @test J[idx.rho_qg_ub[1], idx.rho_qg_ub[1]] == 1.0
    @test J[idx.rho_qg_lb[1], idx.qg[1]] == 0.0
    @test J[idx.rho_qg_ub[1], idx.qg[1]] == 0.0

    # A stated bound is untouched by any of this.
    bounded = ACNetwork(pd_case(buses,
        [pd_gen(1, 1; pg=0.3, qmin=-1.0, qmax=1.0, pmax=2.0, pmin=0.0, cost=(0.0, 1.0, 0.0))],
        branches; name="stated_q_limits"))
    bprob = ACOPFProblem(bounded; silent=true)
    @test JuMP.has_lower_bound(bprob.qg[1])
    bsol = solve!(bprob)
    bidx = kkt_indices(bprob)
    bJ = calc_kkt_jacobian(bprob; sol=bsol)
    @test bJ[bidx.rho_qg_lb[1], bidx.rho_qg_lb[1]] ≈ bsol.qg[1] - bounded.qmin[1]
    @test all(isfinite, kkt(flatten_variables(bsol, bprob), bprob))
end

@testset "Non-finite values PowerDiff cannot model" begin
    # Only the reactive limits model an absent bound. Everywhere else a non-finite
    # value is named, with the row and the field, rather than reaching a factorization.
    buses = [pd_bus(1, 3), pd_bus(2, 1; pd=0.3)]
    branches = [pd_branch(1, 1, 2; br_r=0.01, br_x=0.1, rate_a=2.0)]

    for (field, gen) in (
        (:pmax, pd_gen(1, 1; pmax=Inf, pmin=0.0, cost=(0.0, 1.0, 0.0))),
        (:pmin, pd_gen(1, 1; pmax=2.0, pmin=-Inf, cost=(0.0, 1.0, 0.0))),
    )
        err = try
            ACNetwork(pd_case(buses, [gen], branches))
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin(String(field), sprint(showerror, err))
    end

    err = try
        ACNetwork(pd_case([pd_bus(1, 3; vmax=Inf), pd_bus(2, 1; pd=0.3)],
                          [pd_gen(1, 1; pmax=2.0, pmin=0.0, cost=(0.0, 1.0, 0.0))],
                          branches))
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    @test occursin("vmax", sprint(showerror, err))
end

@testset "The ingest still reproduces MATPOWER's network" begin
    # An oracle that shares no code with the ingest: PowerModels parses the same
    # files independently, and its basic-network PTDF is a function of the topology,
    # the branch susceptances and the reference bus alone. Agreement pins all three.
    PowerModels.silence()
    for c in ("pglib_opf_case14_ieee.m", "pglib_opf_case30_ieee.m", "pglib_opf_case118_ieee.m")
        path = joinpath(PD_PGLIB_DIR, c)
        isfile(path) || continue
        net = PowerDiff.parse_file(path)
        state = DCPowerFlowState(DCNetwork(net), calc_demand_vector(net))
        pm = PowerModels.calc_basic_ptdf_matrix(
            PowerModels.make_basic_network(PowerModels.parse_file(path)))
        @test maximum(abs, ptdf_matrix(state) .- pm) < 1e-8
    end
end
