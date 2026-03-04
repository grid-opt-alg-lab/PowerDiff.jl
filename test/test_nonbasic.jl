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

# =============================================================================
# Non-Basic Network Tests
# =============================================================================
#
# Verifies that PowerModelsDiff works correctly with non-basic networks
# (arbitrary bus/branch/gen IDs, not sequentially renumbered).
#
# case5.m has bus IDs [1,2,3,4,10] — bus 10 maps to sequential index 5.

@testset "Non-Basic Network Support" begin
    raw = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))
    basic = PowerModels.make_basic_network(deepcopy(raw))

    # =================================================================
    # IDMapping
    # =================================================================
    @testset "IDMapping construction" begin
        dc_net = DCNetwork(raw)
        id_map = dc_net.id_map

        @test id_map.bus_ids == [1, 2, 3, 4, 10]
        @test id_map.bus_to_idx[10] == 5
        @test id_map.bus_to_idx[1] == 1
        @test length(id_map.branch_ids) == dc_net.m
        @test length(id_map.gen_ids) == dc_net.k

        # Basic network should produce identity mapping
        dc_net_basic = DCNetwork(basic)
        @test dc_net_basic.id_map.bus_ids == collect(1:dc_net_basic.n)
    end

    # =================================================================
    # DCNetwork from non-basic
    # =================================================================
    @testset "DCNetwork from non-basic network" begin
        dc_nb = DCNetwork(raw)
        dc_b = DCNetwork(basic)

        @test dc_nb.n == dc_b.n
        @test dc_nb.m == dc_b.m
        @test dc_nb.k == dc_b.k

        # Incidence matrices should have same structure
        @test size(dc_nb.A) == size(dc_b.A)
        @test size(dc_nb.G_inc) == size(dc_b.G_inc)

        # Susceptances should match (same branches, same impedances)
        @test isapprox(sort(dc_nb.b), sort(dc_b.b), atol=1e-10)
    end

    # =================================================================
    # DC Power Flow
    # =================================================================
    @testset "DC power flow non-basic" begin
        d_nb = calc_demand_vector(raw)
        d_b = calc_demand_vector(basic)

        # Total demand should match
        @test isapprox(sum(d_nb), sum(d_b), atol=1e-10)

        dc_nb = DCNetwork(raw)
        dc_b = DCNetwork(basic)
        pf_nb = DCPowerFlowState(dc_nb, d_nb)
        pf_b = DCPowerFlowState(dc_b, d_b)

        # Flows should match (same physics, different IDs)
        @test isapprox(sort(abs.(pf_nb.f)), sort(abs.(pf_b.f)), atol=1e-6)
    end

    # =================================================================
    # DC OPF
    # =================================================================
    @testset "DC OPF non-basic" begin
        prob_nb = DCOPFProblem(raw)
        prob_b = DCOPFProblem(basic)

        sol_nb = solve!(prob_nb)
        sol_b = solve!(prob_b)

        # Objectives should be close (same problem, just different ID numbering)
        @test isapprox(sol_nb.objective, sol_b.objective, rtol=0.01)

        # Generation should match (sorted, since gen ordering may differ)
        @test isapprox(sort(sol_nb.pg), sort(sol_b.pg), atol=1e-4)
    end

    # =================================================================
    # DC OPF Sensitivities
    # =================================================================
    @testset "DC OPF sensitivities non-basic" begin
        prob_nb = DCOPFProblem(raw)
        prob_b = DCOPFProblem(basic)
        solve!(prob_nb)
        solve!(prob_b)

        # Compute sensitivities
        dva_dd_nb = calc_sensitivity(prob_nb, :va, :d)
        dva_dd_b = calc_sensitivity(prob_b, :va, :d)

        # row_to_id should contain original bus IDs for non-basic
        @test 10 in dva_dd_nb.row_to_id  # Bus 10 should be in the mapping
        @test dva_dd_nb.row_to_id == [1, 2, 3, 4, 10]

        # col_to_id should also contain original bus IDs (demand is per-bus)
        @test dva_dd_nb.col_to_id == [1, 2, 3, 4, 10]

        # Basic network should have sequential IDs
        @test dva_dd_b.row_to_id == collect(1:prob_b.network.n)

        # Sensitivity matrix values should be close
        # (same physics, but may have different row/col ordering)
        @test isapprox(sort(vec(Matrix(dva_dd_nb))), sort(vec(Matrix(dva_dd_b))), atol=1e-4)

        # Test all operand/parameter combinations produce finite results
        for (op, param) in [(:va, :d), (:pg, :d), (:f, :d), (:lmp, :d), (:psh, :d),
                            (:va, :sw), (:pg, :sw), (:f, :sw), (:lmp, :sw)]
            S = calc_sensitivity(prob_nb, op, param)
            @test all(isfinite, Matrix(S))
        end

        # Generator sensitivity: row_to_id should be gen IDs
        dpg_dd_nb = calc_sensitivity(prob_nb, :pg, :d)
        @test dpg_dd_nb.row_to_id == sort(collect(parse(Int, k) for k in keys(raw["gen"])))

        # Branch sensitivity: row_to_id should be branch IDs
        df_dsw_nb = calc_sensitivity(prob_nb, :f, :sw)
        @test length(df_dsw_nb.row_to_id) == prob_nb.network.m
    end

    # =================================================================
    # DC PF Sensitivities
    # =================================================================
    @testset "DC PF sensitivities non-basic" begin
        dc_nb = DCNetwork(raw)
        d_nb = calc_demand_vector(raw)
        pf_nb = DCPowerFlowState(dc_nb, d_nb)

        dva_dd = calc_sensitivity(pf_nb, :va, :d)
        df_dsw = calc_sensitivity(pf_nb, :f, :sw)

        @test dva_dd.row_to_id == [1, 2, 3, 4, 10]
        @test dva_dd.col_to_id == [1, 2, 3, 4, 10]
        @test all(isfinite, Matrix(dva_dd))
        @test all(isfinite, Matrix(df_dsw))
    end

    # =================================================================
    # ACNetwork from non-basic
    # =================================================================
    @testset "ACNetwork from non-basic network" begin
        ac_nb = ACNetwork(raw)
        ac_b = ACNetwork(basic)

        @test ac_nb.n == ac_b.n
        @test ac_nb.m == ac_b.m
        @test ac_nb.id_map.bus_ids == [1, 2, 3, 4, 10]

        # Admittance matrix should produce same eigenvalues (same physics)
        Y_nb = admittance_matrix(ac_nb)
        Y_b = admittance_matrix(ac_b)
        @test isapprox(sort(abs.(eigvals(Matrix(Y_nb)))),
                       sort(abs.(eigvals(Matrix(Y_b)))), rtol=0.01)
    end

    # =================================================================
    # AC Power Flow
    # =================================================================
    @testset "AC power flow non-basic" begin
        # Solve AC PF using PowerModels on non-basic network
        pf_data_nb = deepcopy(raw)
        PowerModels.compute_ac_pf!(pf_data_nb)
        state_nb = ACPowerFlowState(pf_data_nb)

        # Voltage sensitivity should be finite
        dvm_dp = calc_sensitivity(state_nb, :vm, :p)
        @test all(isfinite, Matrix(dvm_dp))
        @test dvm_dp.row_to_id == [1, 2, 3, 4, 10]

        # Current sensitivity should be finite
        dim_dp = calc_sensitivity(state_nb, :im, :p)
        @test all(isfinite, Matrix(dim_dp))

        # Slack bus voltage sensitivity should be zero
        slack = state_nb.idx_slack
        @test maximum(abs.(Matrix(dvm_dp)[slack, :])) < 1e-10
    end

    # =================================================================
    # AC OPF
    # =================================================================
    @testset "AC OPF non-basic" begin
        prob_nb = ACOPFProblem(raw)
        sol_nb = solve!(prob_nb)

        # Solution should be feasible
        @test sol_nb.objective > 0
        @test all(isfinite, sol_nb.vm)
        @test all(isfinite, sol_nb.va)

        # Sensitivity checks
        dvm_dsw = calc_sensitivity(prob_nb, :vm, :sw)
        @test all(isfinite, Matrix(dvm_dsw))
        @test dvm_dsw.row_to_id == [1, 2, 3, 4, 10]

        dva_dsw = calc_sensitivity(prob_nb, :va, :sw)
        @test all(isfinite, Matrix(dva_dsw))

        dpg_dsw = calc_sensitivity(prob_nb, :pg, :sw)
        @test all(isfinite, Matrix(dpg_dsw))
        @test dpg_dsw.row_to_id == sort(collect(parse(Int, k) for k in keys(raw["gen"])))
    end

    # =================================================================
    # Cross-validation: sensitivity values match between basic and non-basic
    # =================================================================
    @testset "Sensitivity values match basic vs non-basic" begin
        prob_nb = DCOPFProblem(raw)
        prob_b = DCOPFProblem(basic)
        solve!(prob_nb)
        solve!(prob_b)

        # For each parameter, compare sensitivity values
        # The matrices may have different row/col ordering but same physics
        for (op, param) in [(:va, :d), (:pg, :d), (:f, :d), (:lmp, :d)]
            S_nb = Matrix(calc_sensitivity(prob_nb, op, param))
            S_b = Matrix(calc_sensitivity(prob_b, op, param))

            # Frobenius norm comparison (sorted singular values)
            sv_nb = sort(svd(S_nb).S, rev=true)
            sv_b = sort(svd(S_b).S, rev=true)
            @test isapprox(sv_nb, sv_b, atol=1e-4)
        end
    end

    # =================================================================
    # id_to_row / id_to_col round-trip
    # =================================================================
    @testset "id_to_row / id_to_col round-trip" begin
        prob_nb = DCOPFProblem(raw)
        solve!(prob_nb)

        S = calc_sensitivity(prob_nb, :va, :d)
        for (i, id) in enumerate(S.row_to_id)
            @test S.id_to_row[id] == i
        end
        for (j, id) in enumerate(S.col_to_id)
            @test S.id_to_col[id] == j
        end

        # Also check generator-indexed sensitivity
        Spg = calc_sensitivity(prob_nb, :pg, :d)
        for (i, id) in enumerate(Spg.row_to_id)
            @test Spg.id_to_row[id] == i
        end
    end

    # =================================================================
    # Element-wise cross-validation using ID mappings
    # =================================================================
    @testset "Element-wise cross-validation" begin
        prob_nb = DCOPFProblem(raw)
        prob_b = DCOPFProblem(basic)
        solve!(prob_nb)
        solve!(prob_b)

        S_b = calc_sensitivity(prob_b, :va, :d)
        S_nb = calc_sensitivity(prob_nb, :va, :d)

        # Pick bus ID 1 (exists in both basic and non-basic)
        bus_id = 1
        row_b = S_b.id_to_row[bus_id]
        col_b = S_b.id_to_col[bus_id]
        row_nb = S_nb.id_to_row[bus_id]
        col_nb = S_nb.id_to_col[bus_id]
        @test isapprox(S_b[row_b, col_b], S_nb[row_nb, col_nb], atol=1e-6)

        # Check generator sensitivities element-wise
        Spg_b = calc_sensitivity(prob_b, :pg, :d)
        Spg_nb = calc_sensitivity(prob_nb, :pg, :d)

        gen_id = first(Spg_b.row_to_id)
        if gen_id in Spg_nb.row_to_id
            row_b = Spg_b.id_to_row[gen_id]
            row_nb = Spg_nb.id_to_row[gen_id]
            col_b = Spg_b.id_to_col[bus_id]
            col_nb = Spg_nb.id_to_col[bus_id]
            @test isapprox(Spg_b[row_b, col_b], Spg_nb[row_nb, col_nb], atol=1e-4)
        end
    end

    # =================================================================
    # Stored ref on network types
    # =================================================================
    @testset "Stored ref field" begin
        dc_nb = DCNetwork(raw)
        @test !isnothing(dc_nb.ref)
        @test haskey(dc_nb.ref, :bus)

        ac_nb = ACNetwork(raw)
        @test !isnothing(ac_nb.ref)
        @test haskey(ac_nb.ref, :bus)

        # Programmatic constructor has ref=nothing
        n, m, k = 2, 1, 1
        A = sparse([1.0 -1.0])
        G_inc = sparse(reshape([1.0, 0.0], 2, 1))
        b = [-10.0]
        dc_prog = DCNetwork(n, m, k, A, G_inc, b)
        @test isnothing(dc_prog.ref)
    end

    # =================================================================
    # IDMapping shunt support
    # =================================================================
    @testset "IDMapping shunt fields" begin
        dc_nb = DCNetwork(raw)
        id_map = dc_nb.id_map

        @test isa(id_map.shunt_ids, Vector{Int})
        @test isa(id_map.shunt_to_idx, Dict{Int,Int})
        # shunt_ids should be sorted
        @test issorted(id_map.shunt_ids)
    end

    # =================================================================
    # calc_demand_vector from DCNetwork
    # =================================================================
    @testset "calc_demand_vector from DCNetwork" begin
        dc_nb = DCNetwork(raw)
        d1 = calc_demand_vector(raw)
        d2 = calc_demand_vector(dc_nb)
        @test isapprox(d1, d2, atol=1e-10)
    end
end
