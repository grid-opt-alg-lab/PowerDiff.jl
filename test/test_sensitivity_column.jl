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
# Tests for calc_sensitivity_column
# =============================================================================
#
# Verifies calc_sensitivity_column matches the corresponding column of the
# full calc_sensitivity matrix for all formulations (DC PF, DC OPF, AC PF,
# AC OPF), including cached and uncached paths, non-basic networks, and
# error handling.

@testset "calc_sensitivity_column" begin

    # =========================================================================
    # DC Power Flow
    # =========================================================================
    @testset "DC PF" begin
        pm_data = load_test_case("case5.m")
        @test !isnothing(pm_data)
        net = DCNetwork(pm_data)
        d = calc_demand_vector(pm_data)
        state = DCPowerFlowState(net, d)

        for (op, param) in [(:va, :d), (:f, :d), (:va, :sw), (:f, :sw), (:va, :b), (:f, :b)]
            S = calc_sensitivity(state, op, param)
            # Test first, middle, and last columns
            test_cols = [1, size(S, 2)]
            if size(S, 2) >= 3
                push!(test_cols, div(size(S, 2), 2) + 1)
            end
            for j in test_cols
                col_id = S.col_to_id[j]
                col = calc_sensitivity_column(state, op, param, col_id)
                @test col ≈ Matrix(S)[:, j] atol=1e-12
            end
        end
    end

    # =========================================================================
    # DC OPF — uncached (fresh problem per combo)
    # =========================================================================
    @testset "DC OPF (uncached)" begin
        pm_data = load_test_case("case5.m")
        @test !isnothing(pm_data)

        for (op, param) in [(:lmp, :d), (:va, :sw), (:pg, :cq), (:f, :cl), (:psh, :fmax), (:lmp, :b)]
            # Reference full matrix
            net = DCNetwork(pm_data)
            d = calc_demand_vector(pm_data)
            prob = DCOPFProblem(net, d)
            solve!(prob)
            S = calc_sensitivity(prob, op, param)

            # Test first, middle, and last column with fresh (uncached) problems
            test_cols = [1, size(S, 2)]
            if size(S, 2) >= 3
                push!(test_cols, div(size(S, 2), 2) + 1)
            end
            for j in test_cols
                col_id = S.col_to_id[j]
                net2 = DCNetwork(pm_data)
                d2 = calc_demand_vector(pm_data)
                prob2 = DCOPFProblem(net2, d2)
                solve!(prob2)

                col = calc_sensitivity_column(prob2, op, param, col_id)
                @test col ≈ Matrix(S)[:, j] atol=1e-10
            end
        end
    end

    # =========================================================================
    # DC OPF — load shedding column (psh with active shedding)
    # =========================================================================
    @testset "DC OPF (psh shedding)" begin
        result = _make_case14_shedding()
        if !isnothing(result)
            prob = DCOPFProblem(result.dc_net, result.d)
            solve!(prob)
            S = calc_sensitivity(prob, :psh, :d)
            for j in [1, size(S, 2)]
                col_id = S.col_to_id[j]
                col = calc_sensitivity_column(prob, :psh, :d, col_id)
                @test col ≈ Matrix(S)[:, j] atol=1e-10
            end
        end
    end

    # =========================================================================
    # DC OPF — cached (call full matrix first, then column)
    # =========================================================================
    @testset "DC OPF (cached)" begin
        pm_data = load_test_case("case5.m")
        @test !isnothing(pm_data)
        net = DCNetwork(pm_data)
        d = calc_demand_vector(pm_data)
        prob = DCOPFProblem(net, d)
        solve!(prob)

        for (op, param) in [(:lmp, :d), (:pg, :sw), (:f, :cq)]
            S = calc_sensitivity(prob, op, param)
            # Now cache is populated — column should use fast path
            for j in [1, size(S, 2)]
                col_id = S.col_to_id[j]
                col = calc_sensitivity_column(prob, op, param, col_id)
                @test col ≈ Matrix(S)[:, j] atol=1e-12
            end
        end
    end

    # =========================================================================
    # AC Power Flow
    # =========================================================================
    @testset "AC PF" begin
        pm_data = load_test_case("case5.m")
        @test !isnothing(pm_data)
        ac_net = ACNetwork(pm_data)
        ac_state = ACPowerFlowState(pm_data)

        for (op, param) in [(:vm, :p), (:va, :q), (:im, :p), (:f, :q)]
            S = calc_sensitivity(ac_state, op, param)
            col_id = S.col_to_id[1]
            col = calc_sensitivity_column(ac_state, op, param, col_id)
            @test col ≈ Matrix(S)[:, 1] atol=1e-12
        end

        # Topology sensitivities
        for (op, param) in [(:vm, :g), (:va, :b)]
            S = calc_sensitivity(ac_state, op, param)
            col_id = S.col_to_id[1]
            col = calc_sensitivity_column(ac_state, op, param, col_id)
            @test col ≈ Matrix(S)[:, 1] atol=1e-12
        end

        # Parameter transform: :d → -:p
        S = calc_sensitivity(ac_state, :vm, :d)
        col_id = S.col_to_id[1]
        col = calc_sensitivity_column(ac_state, :vm, :d, col_id)
        @test col ≈ Matrix(S)[:, 1] atol=1e-12
    end

    # =========================================================================
    # AC OPF — uncached
    # =========================================================================
    @testset "AC OPF (uncached)" begin
        pm_data = load_test_case("case5.m")
        @test !isnothing(pm_data)

        for (op, param) in [(:lmp, :d), (:vm, :sw), (:pg, :cq)]
            # Fresh problem each time
            prob = ACOPFProblem(pm_data)

            S = calc_sensitivity(prob, op, param)
            col_id = S.col_to_id[1]

            # New fresh problem — no cache
            prob2 = ACOPFProblem(pm_data)
            col = calc_sensitivity_column(prob2, op, param, col_id)
            @test col ≈ Matrix(S)[:, 1] atol=1e-8
        end
    end

    # =========================================================================
    # AC OPF — cached
    # =========================================================================
    @testset "AC OPF (cached)" begin
        pm_data = load_test_case("case5.m")
        @test !isnothing(pm_data)
        prob = ACOPFProblem(pm_data)

        for (op, param) in [(:lmp, :d), (:va, :sw), (:qg, :fmax), (:qlmp, :d)]
            S = calc_sensitivity(prob, op, param)
            for j in [1, size(S, 2)]
                col_id = S.col_to_id[j]
                col = calc_sensitivity_column(prob, op, param, col_id)
                @test col ≈ Matrix(S)[:, j] atol=1e-12
            end
        end
    end

    # =========================================================================
    # Non-basic network (arbitrary element IDs)
    # =========================================================================
    @testset "Non-basic network" begin
        pm_data = load_raw_case("case5.m")
        @test !isnothing(pm_data)

        # case5.m has bus IDs [1, 2, 3, 4, 10]
        net = DCNetwork(pm_data)
        d = calc_demand_vector(pm_data)
        prob = DCOPFProblem(net, d)
        solve!(prob)

        S = calc_sensitivity(prob, :lmp, :d)
        # Use an actual bus ID (should include non-sequential ID like 10)
        for j in 1:size(S, 2)
            col_id = S.col_to_id[j]
            col = calc_sensitivity_column(prob, :lmp, :d, col_id)
            @test col ≈ Matrix(S)[:, j] atol=1e-10
        end
    end

    # =========================================================================
    # Error handling
    # =========================================================================
    @testset "Error handling" begin
        pm_data = load_test_case("case5.m")
        @test !isnothing(pm_data)
        net = DCNetwork(pm_data)
        d = calc_demand_vector(pm_data)
        prob = DCOPFProblem(net, d)
        solve!(prob)

        # Invalid element ID
        @test_throws ArgumentError calc_sensitivity_column(prob, :lmp, :d, 9999)

        # Invalid operand/parameter combo
        @test_throws ArgumentError calc_sensitivity_column(prob, :vm, :d, 1)
    end
end
