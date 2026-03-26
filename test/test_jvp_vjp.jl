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
# JVP / VJP Tests
# =============================================================================
#
# Verifies JVP/VJP with ID-aware Dict I/O against direct matrix-vector
# multiplication for DC PF, DC OPF, and AC PF (including complex-valued
# sensitivities). Also tests the transpose identity dot(w, S*v) = dot(S'*w, v),
# round-trip dict_to_vec/vec_to_dict, and error handling for invalid IDs.

@testset "JVP / VJP" begin
    raw = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))
    basic = PowerModels.make_basic_network(deepcopy(raw))

    # =================================================================
    # Basic network JVP
    # =================================================================
    @testset "Basic network JVP matches S * v" begin
        prob = DCOPFProblem(basic)
        solve!(prob)
        S = calc_sensitivity(prob, :lmp, :d)

        v = randn(size(S, 2))
        result_dict = jvp(S, v)
        result_vec = S * v

        # Dict keys should be sequential 1:n for basic network
        @test sort(collect(keys(result_dict))) == collect(1:size(S, 1))

        # Values should match S * v element-wise
        for (id, val) in result_dict
            @test val ≈ result_vec[S.id_to_row[id]]
        end
    end

    # =================================================================
    # Non-basic network JVP with Dict input
    # =================================================================
    @testset "Non-basic JVP with Dict input" begin
        prob_nb = DCOPFProblem(raw)
        solve!(prob_nb)
        S = calc_sensitivity(prob_nb, :lmp, :d)

        # Perturb demand at bus 10 only
        δp = Dict(10 => 0.1)
        result = jvp(S, δp)

        # Result should have original bus IDs as keys
        @test sort(collect(keys(result))) == [1, 2, 3, 4, 10]

        # Should match extracting column for bus 10, scaled by 0.1
        col_idx = S.id_to_col[10]
        for (id, val) in result
            row_idx = S.id_to_row[id]
            @test val ≈ S.matrix[row_idx, col_idx] * 0.1
        end
    end

    # =================================================================
    # VJP correctness
    # =================================================================
    @testset "VJP matches S' * w" begin
        prob = DCOPFProblem(basic)
        solve!(prob)
        S = calc_sensitivity(prob, :pg, :d)

        w = randn(size(S, 1))
        result_dict = vjp(S, w)
        result_vec = S.matrix' * w

        @test sort(collect(keys(result_dict))) == sort(collect(S.col_to_id))
        for (id, val) in result_dict
            @test val ≈ result_vec[S.id_to_col[id]]
        end
    end

    # =================================================================
    # Non-basic VJP with Dict input
    # =================================================================
    @testset "Non-basic VJP with Dict input" begin
        prob_nb = DCOPFProblem(raw)
        solve!(prob_nb)
        S = calc_sensitivity(prob_nb, :lmp, :d)

        # Adjoint seed at bus 10
        δy = Dict(10 => 1.0)
        result = vjp(S, δy)

        # Should match extracting row for bus 10 (transposed)
        row_idx = S.id_to_row[10]
        for (id, val) in result
            col_idx = S.id_to_col[id]
            @test val ≈ S.matrix[row_idx, col_idx]
        end
    end

    # =================================================================
    # Transpose identity: dot(w, S*v) ≈ dot(vjp_vec, v)
    # =================================================================
    @testset "Transpose identity" begin
        prob = DCOPFProblem(basic)
        solve!(prob)
        S = calc_sensitivity(prob, :f, :d)

        v = randn(size(S, 2))
        w = randn(size(S, 1))

        lhs = dot(w, S * v)
        vjp_result = vjp(S, w)
        vjp_vec = [vjp_result[S.col_to_id[j]] for j in 1:size(S, 2)]
        rhs = dot(vjp_vec, v)
        @test lhs ≈ rhs atol=1e-10
    end

    # =================================================================
    # Error handling
    # =================================================================
    @testset "Error handling" begin
        prob = DCOPFProblem(basic)
        solve!(prob)
        S = calc_sensitivity(prob, :lmp, :d)

        # Unknown ID throws ArgumentError
        @test_throws ArgumentError jvp(S, Dict(9999 => 1.0))
        @test_throws ArgumentError vjp(S, Dict(9999 => 1.0))

        # Wrong-length vector throws DimensionMismatch
        @test_throws DimensionMismatch jvp(S, randn(size(S, 2) + 1))
        @test_throws DimensionMismatch vjp(S, randn(size(S, 1) + 1))

        # Invalid axis throws ArgumentError
        @test_throws ArgumentError dict_to_vec(S, Dict(1 => 1.0), :bad)
        @test_throws ArgumentError vec_to_dict(S, randn(size(S, 1)), :bad)

        # Wrong-length vector in vec_to_dict
        @test_throws DimensionMismatch vec_to_dict(S, randn(size(S, 1) + 1), :row)
    end

    # =================================================================
    # Round-trip: vec_to_dict ∘ dict_to_vec == identity
    # =================================================================
    @testset "Round-trip dict_to_vec / vec_to_dict" begin
        prob_nb = DCOPFProblem(raw)
        solve!(prob_nb)
        S = calc_sensitivity(prob_nb, :va, :d)

        d_orig = Dict(1 => 0.5, 10 => -0.3)
        v = dict_to_vec(S, d_orig, :col)
        d_round = vec_to_dict(S, v, :col)

        # Original keys should match
        for (id, val) in d_orig
            @test d_round[id] ≈ val
        end
        # Non-specified keys should be zero
        for id in S.col_to_id
            if !haskey(d_orig, id)
                @test d_round[id] ≈ 0.0
            end
        end
    end

    # =================================================================
    # Complex sensitivity (AC PF :v operand)
    # =================================================================
    @testset "Complex sensitivity JVP/VJP" begin
        pf_data = deepcopy(basic)
        PowerModels.compute_ac_pf!(pf_data)
        state = ACPowerFlowState(pf_data)

        S = calc_sensitivity(state, :v, :p)

        # JVP with Dict input
        δp = Dict(1 => 0.01)
        result = jvp(S, δp)
        @test eltype(values(result)) <: Complex

        col_idx = S.id_to_col[1]
        for (id, val) in result
            row_idx = S.id_to_row[id]
            @test val ≈ S.matrix[row_idx, col_idx] * 0.01
        end

        # VJP with Dict input (uses adjoint = conjugate transpose)
        δy = Dict(2 => 1.0 + 0.0im)
        vjp_result = vjp(S, δy)
        row_idx = S.id_to_row[2]
        for (id, val) in vjp_result
            col_idx = S.id_to_col[id]
            @test val ≈ conj(S.matrix[row_idx, col_idx])
        end
    end

    # =================================================================
    # Branch-indexed sensitivities (:f w.r.t. :sw)
    # =================================================================
    @testset "Branch-indexed JVP" begin
        prob_nb = DCOPFProblem(raw)
        solve!(prob_nb)
        S = calc_sensitivity(prob_nb, :f, :sw)

        # row_to_id = branch IDs, col_to_id = branch IDs
        branch_ids = S.col_to_id
        δsw = Dict(branch_ids[1] => 0.01)
        result = jvp(S, δsw)

        @test sort(collect(keys(result))) == sort(S.row_to_id)
        col_idx = S.id_to_col[branch_ids[1]]
        for (id, val) in result
            @test val ≈ S.matrix[S.id_to_row[id], col_idx] * 0.01
        end
    end

    # =================================================================
    # Generator-indexed sensitivities (:pg w.r.t. :d)
    # =================================================================
    @testset "Generator-indexed JVP" begin
        prob_nb = DCOPFProblem(raw)
        solve!(prob_nb)
        S = calc_sensitivity(prob_nb, :pg, :d)

        # row_to_id = gen IDs, col_to_id = bus IDs
        @test 10 in S.col_to_id  # bus 10

        δd = Dict(10 => 0.1)
        result = jvp(S, δd)
        @test sort(collect(keys(result))) == sort(S.row_to_id)
    end

    # =================================================================
    # Empty Dict → all-zero result
    # =================================================================
    @testset "Empty Dict input" begin
        prob = DCOPFProblem(basic)
        solve!(prob)
        S = calc_sensitivity(prob, :lmp, :d)

        result = jvp(S, Dict{Int,Float64}())
        @test all(v ≈ 0.0 for v in values(result))

        result_vjp = vjp(S, Dict{Int,Float64}())
        @test all(v ≈ 0.0 for v in values(result_vjp))
    end
end
