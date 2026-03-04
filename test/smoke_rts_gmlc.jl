# =============================================================================
# Smoke Test: RTS_GMLC (Non-Basic Network with Realistic Scale)
# =============================================================================
#
# Validates non-basic network support against RTS_GMLC:
#   73 buses (IDs 101-325), 120 branches, 96 generators, 51 loads
#
# Run manually:
#   julia --project=. test/smoke_rts_gmlc.jl
#
# NOT included in Pkg.test() because:
#   - RTS_GMLC.m lives in PowerSystems/PowerSystemCaseBuilder, not PowerModels
#   - File path depends on installed package version
#   - AC OPF + ForwardDiff KKT Jacobian is slow (~30s+)
#   - DC line workaround required (PowerModels limitation)

using Test
using LinearAlgebra
using PowerModelsDiff
using PowerModels

PowerModels.silence()

# ─────────────────────────────────────────────────────────────────────────────
# Locate RTS_GMLC.m
# ─────────────────────────────────────────────────────────────────────────────
const RTS_PATHS = [
    joinpath(dirname(dirname(pathof(PowerModels))), "test", "data", "matpower", "RTS_GMLC.m"),
    # PowerSystems / PowerSystemCaseBuilder fallbacks
    filter(isfile, [
        joinpath(d, "data", "matpower", "RTS_GMLC.m")
        for d in readdir(joinpath(homedir(), ".julia", "packages", "PowerSystemCaseBuilder"); join=true)
        if isdir(d)
    ])...,
    filter(isfile, [
        joinpath(d, "data", "matpower", "RTS_GMLC.m")
        for d in readdir(joinpath(homedir(), ".julia", "packages", "PowerSystems"); join=true)
        if isdir(d)
    ])...,
]

const RTS_PATH = let found = filter(isfile, RTS_PATHS)
    if isempty(found)
        error("""
        RTS_GMLC.m not found. Install one of:
            PowerSystems.jl    (Pkg.add("PowerSystems"))
            PowerSystemCaseBuilder.jl (Pkg.add("PowerSystemCaseBuilder"))
        """)
    end
    first(found)
end

println("Using RTS_GMLC.m from: $RTS_PATH")

# ─────────────────────────────────────────────────────────────────────────────
# Load network
# ─────────────────────────────────────────────────────────────────────────────
raw = PowerModels.parse_file(RTS_PATH)

# Verify this is truly non-basic (bus IDs are not 1:n)
bus_ids = sort([raw["bus"][k]["bus_i"] for k in keys(raw["bus"])])
@assert bus_ids != collect(1:length(bus_ids)) "RTS_GMLC should have non-sequential bus IDs"
println("Bus IDs: $(bus_ids[1])..$(bus_ids[end]) ($(length(bus_ids)) buses)")
println("Branches: $(length(raw["branch"])), Generators: $(length(raw["gen"])), Loads: $(length(raw["load"]))")

@testset "RTS_GMLC Smoke Tests" begin

    # =================================================================
    # DCNetwork
    # =================================================================
    @testset "DCNetwork construction" begin
        dc_net = DCNetwork(raw)
        @test dc_net.n == length(bus_ids)
        @test dc_net.m == length(raw["branch"])
        # build_ref() filters inactive generators (158 total, 96 active)
        @test dc_net.k == length(dc_net.id_map.gen_ids)
        @test dc_net.k > 0

        # IDMapping should preserve original bus IDs
        @test dc_net.id_map.bus_ids == bus_ids
        @test all(id -> haskey(dc_net.id_map.bus_to_idx, id), bus_ids)

        # Sequential indices should be 1:n
        @test sort(collect(values(dc_net.id_map.bus_to_idx))) == collect(1:dc_net.n)
    end

    # =================================================================
    # DC Power Flow
    # =================================================================
    @testset "DC power flow" begin
        dc_net = DCNetwork(raw)
        d = calc_demand_vector(raw)
        @test length(d) == dc_net.n
        @test sum(d) > 0  # nonzero demand

        pf = DCPowerFlowState(dc_net, d)
        @test all(isfinite, pf.va)
        @test all(isfinite, pf.f)

        # Sensitivity checks
        dva_dd = calc_sensitivity(pf, :va, :d)
        @test all(isfinite, Matrix(dva_dd))
        @test dva_dd.row_to_id == bus_ids
        @test dva_dd.col_to_id == bus_ids

        df_dsw = calc_sensitivity(pf, :f, :sw)
        @test all(isfinite, Matrix(df_dsw))
        @test length(df_dsw.row_to_id) == dc_net.m
    end

    # =================================================================
    # DC OPF
    # =================================================================
    local dc_prob  # share across DC testsets
    @testset "DC OPF solve" begin
        dc_prob = DCOPFProblem(raw)
        sol = solve!(dc_prob)
        @test sol.objective > 0
        @test all(isfinite, sol.pg)
        @test all(isfinite, sol.va)
        @test all(isfinite, sol.f)
        println("  DC OPF objective: $(round(sol.objective, digits=2))")
    end

    @testset "DC OPF sensitivities" begin
        combos = [
            (:va, :d), (:pg, :d), (:f, :d), (:lmp, :d), (:psh, :d),
            (:va, :sw), (:pg, :sw), (:f, :sw), (:lmp, :sw),
            (:pg, :cq), (:pg, :cl), (:lmp, :cq), (:lmp, :cl),
            (:f, :fmax), (:pg, :fmax), (:lmp, :fmax),
            (:va, :b), (:f, :b), (:pg, :b), (:lmp, :b),
        ]
        @testset "d$(op)/d$(param)" for (op, param) in combos
            S = calc_sensitivity(dc_prob, op, param)
            @test all(isfinite, Matrix(S))
        end
    end

    @testset "DC sensitivity ID mapping" begin
        dva_dd = calc_sensitivity(dc_prob, :va, :d)
        @test dva_dd.row_to_id == bus_ids
        @test dva_dd.col_to_id == bus_ids

        # gen/branch IDs come from build_ref (active elements only)
        dpg_dd = calc_sensitivity(dc_prob, :pg, :d)
        @test dpg_dd.row_to_id == dc_prob.network.id_map.gen_ids

        df_dsw = calc_sensitivity(dc_prob, :f, :sw)
        @test df_dsw.row_to_id == dc_prob.network.id_map.branch_ids
        @test df_dsw.col_to_id == dc_prob.network.id_map.branch_ids
    end

    # =================================================================
    # ACNetwork (requires emptying dcline — PowerModels limitation)
    # =================================================================
    raw_ac = deepcopy(raw)
    if !isempty(raw_ac["dcline"])
        println("  Emptying $(length(raw_ac["dcline"])) DC line(s) for AC compatibility")
        empty!(raw_ac["dcline"])
    end

    @testset "ACNetwork construction" begin
        ac_net = ACNetwork(raw_ac)
        @test ac_net.n == length(bus_ids)
        @test ac_net.m == length(raw_ac["branch"])
        @test ac_net.id_map.bus_ids == bus_ids
    end

    # =================================================================
    # AC Power Flow
    # =================================================================
    @testset "AC power flow" begin
        pf_data = deepcopy(raw_ac)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        @test all(isfinite, abs.(state.v))
        @test all(v -> 0.8 < abs(v) < 1.2, state.v)  # reasonable voltage magnitudes

        dvm_dp = calc_sensitivity(state, :vm, :p)
        @test all(isfinite, Matrix(dvm_dp))
        @test dvm_dp.row_to_id == bus_ids

        dim_dp = calc_sensitivity(state, :im, :p)
        @test all(isfinite, Matrix(dim_dp))

        # Slack bus voltage sensitivity should be near zero
        slack = state.idx_slack
        @test maximum(abs.(Matrix(dvm_dp)[slack, :])) < 1e-10
    end

    # =================================================================
    # AC OPF
    # =================================================================
    @testset "AC OPF" begin
        ac_prob = ACOPFProblem(raw_ac)
        sol = solve!(ac_prob)
        @test sol.objective > 0
        @test all(isfinite, sol.vm)
        @test all(isfinite, sol.va)
        @test all(isfinite, sol.pg)
        @test all(isfinite, sol.qg)
        println("  AC OPF objective: $(round(sol.objective, digits=2))")

        # Switching sensitivities
        @testset "d$(op)/dsw" for op in (:vm, :va, :pg, :qg)
            S = calc_sensitivity(ac_prob, op, :sw)
            @test all(isfinite, Matrix(S))
        end

        # ID mapping check
        dvm_dsw = calc_sensitivity(ac_prob, :vm, :sw)
        @test dvm_dsw.row_to_id == bus_ids

        dpg_dsw = calc_sensitivity(ac_prob, :pg, :sw)
        @test dpg_dsw.row_to_id == ac_prob.network.id_map.gen_ids
    end

    # =================================================================
    # Cross-validate: DC basic vs non-basic (SVD comparison)
    # =================================================================
    @testset "DC basic vs non-basic SVD match" begin
        basic = PowerModels.make_basic_network(deepcopy(raw))
        prob_nb = DCOPFProblem(raw)
        prob_b = DCOPFProblem(basic)
        sol_nb = solve!(prob_nb)
        sol_b = solve!(prob_b)

        # Objectives should be close
        @test isapprox(sol_nb.objective, sol_b.objective, rtol=0.01)

        # Sensitivity matrices should have matching singular value spectra
        @testset "SVD d$(op)/d$(param)" for (op, param) in [(:va, :d), (:pg, :d), (:f, :d), (:lmp, :d)]
            S_nb = Matrix(calc_sensitivity(prob_nb, op, param))
            S_b = Matrix(calc_sensitivity(prob_b, op, param))
            sv_nb = sort(svd(S_nb).S, rev=true)
            sv_b = sort(svd(S_b).S, rev=true)
            @test isapprox(sv_nb, sv_b, atol=1e-4)
        end
    end

end

println("\nRTS_GMLC smoke tests completed.")
