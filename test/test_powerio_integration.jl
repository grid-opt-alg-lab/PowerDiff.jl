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
    # refused here. This is the property that keeps the two in step.
    for token in (:psse34, :psse35, :pandapower, :pypsa, :pslf, :epc, :pwb,
                  :gridfm, :goc3, :surge, :opfdata, Symbol("pandapower-json"))
        @test PowerDiff._format_token(token) == lowercase(String(token))
    end

    # Two hints must agree, and the alias layer is what lets them.
    @test PowerDiff._powerio_format_hint(:matpower, ".m") == "matpower"
    @test_throws ArgumentError PowerDiff._powerio_format_hint(:matpower, :psse)

    # A bare `json` names a container, not a reader.
    @test_throws ArgumentError PowerDiff._format_token(:json)
    @test_throws ArgumentError PowerDiff._format_token(".json")

    # Distribution formats parse to a MulticonductorNetwork, which PowerDiff does not
    # model; the refusal says so rather than failing three frames down.
    for token in (:dss, :opendss, :pmd, :bmopf)
        err = try
            PowerDiff._format_token(token)
            nothing
        catch caught
            caught
        end
        @test err isa ArgumentError
        @test occursin("distribution", sprint(showerror, err))
    end

    @test_throws ArgumentError PowerDiff._format_token("")
end

@testset "PowerIO ingest is one pass per network" begin
    net = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)

    # Memoized on the network: DCNetwork and ACNetwork built from one parsed network
    # read the same tables, so they cannot describe different cases.
    tables = PowerDiff._network_data(net)
    @test PowerDiff._network_data(net) === tables

    dc = DCNetwork(net)
    ac = ACNetwork(net)
    @test dc.n == ac.n
    @test dc.m == ac.m
    @test dc.id_map.bus_ids == ac.id_map.bus_ids
    @test dc.id_map.branch_ids == ac.id_map.branch_ids

    # A separately parsed network is a separate entry, and agrees.
    other = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)
    other_tables = PowerDiff._network_data(other)
    @test other_tables !== tables
    @test [b.bus_i for b in other_tables.bus] == [b.bus_i for b in tables.bus]
    @test [g.pmax for g in other_tables.gen] == [g.pmax for g in tables.gen]
end

@testset "PowerIO findings travel with the network" begin
    net = PowerDiff.parse_file("pglib_opf_case14_ieee.m"; library=:pglib)
    findings = network_findings(net)

    @test findings isa NamedTuple
    @test propertynames(findings) == (:reader, :normalize)
    @test findings.reader isa AbstractVector
    @test findings.normalize isa AbstractVector
    @test findings.reader == PowerIO.warnings(net)

    # Every finding reads `CODE: message`, the shape a consumer branches on.
    for line in vcat(collect(findings.reader), collect(findings.normalize))
        @test occursin(": ", String(line))
    end

    # Reading findings does not disturb the tables it shares an ingest with.
    @test PowerDiff._network_data(net) === PowerDiff._powerio_ingest(net).tables
    @test network_findings(net).normalize == findings.normalize
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
