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
# Efficient VJP/JVP Through KKT Tests
# =============================================================================
#
# Verifies that vjp(prob, :op, :param, adj) and jvp(prob, :op, :param, tang)
# match the materialized Sensitivity matrix path.

@testset "KKT VJP/JVP" begin
    basic = PowerModels.make_basic_network(
        PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m")))
    raw = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))

    # =================================================================
    # DC OPF
    # =================================================================
    @testset "DC OPF" begin
        prob = DCOPFProblem(basic)
        solve!(prob)

        @testset "VJP matches S' * w — all (op, param)" begin
            for (op, param) in [(:lmp, :d), (:pg, :d), (:f, :sw), (:va, :cq),
                                (:psh, :cl), (:lmp, :fmax), (:f, :b)]
                S = calc_sensitivity(prob, op, param)
                w = randn(size(S, 1))
                expected = S.matrix' * w
                result = vjp(prob, op, param, w)
                @test result ≈ expected atol=1e-10
            end
        end

        @testset "JVP matches S * v — all (op, param)" begin
            for (op, param) in [(:lmp, :d), (:pg, :sw), (:f, :cq), (:va, :cl),
                                (:psh, :fmax), (:lmp, :b)]
                S = calc_sensitivity(prob, op, param)
                v = randn(size(S, 2))
                expected = S.matrix * v
                result = jvp(prob, op, param, v)
                @test result ≈ expected atol=1e-10
            end
        end

        @testset "Transpose identity" begin
            v = randn(5)  # n_buses = 5
            w = randn(5)
            lhs = dot(w, jvp(prob, :lmp, :d, v))
            rhs = dot(v, vjp(prob, :lmp, :d, w))
            @test lhs ≈ rhs atol=1e-10
        end

        @testset "Dict input/output" begin
            adj_dict = Dict(1 => 1.0, 3 => -0.5)
            result = vjp(prob, :lmp, :d, adj_dict)
            @test result isa Dict{Int,Float64}
            @test sort(collect(keys(result))) == [1, 2, 3, 4, 5]

            tang_dict = Dict(2 => 0.1)
            result = jvp(prob, :lmp, :d, tang_dict)
            @test result isa Dict{Int,Float64}
            @test sort(collect(keys(result))) == [1, 2, 3, 4, 5]
        end

        @testset "Fast path uses cache" begin
            # Force cache population
            prob2 = DCOPFProblem(basic)
            solve!(prob2)
            S = calc_sensitivity(prob2, :lmp, :d)
            @test !isnothing(prob2.cache.dz_dd)

            # VJP/JVP should use cached matrix
            w = randn(size(S, 1))
            v = randn(size(S, 2))
            @test vjp(prob2, :lmp, :d, w) ≈ S.matrix' * w atol=1e-10
            @test jvp(prob2, :lmp, :d, v) ≈ S.matrix * v atol=1e-10
        end

        @testset "In-place vjp!/jvp!" begin
            prob3 = DCOPFProblem(basic); solve!(prob3)
            S = calc_sensitivity(prob3, :lmp, :d)
            w = randn(size(S, 1))
            v = randn(size(S, 2))

            work = zeros(kkt_dims(prob3))
            out_vjp = zeros(size(S, 2))
            out_jvp = zeros(size(S, 1))

            # With explicit workspace
            vjp!(out_vjp, prob3, :lmp, :d, w; work=work)
            @test out_vjp ≈ S.matrix' * w atol=1e-10

            jvp!(out_jvp, prob3, :lmp, :d, v; work=work)
            @test out_jvp ≈ S.matrix * v atol=1e-10

            # Without workspace (auto-allocates)
            w2 = randn(size(S, 1))
            vjp!(out_vjp, prob3, :lmp, :d, w2)
            @test out_vjp ≈ S.matrix' * w2 atol=1e-10

            # Verify repeated calls reuse buffers correctly
            vjp!(out_vjp, prob3, :lmp, :d, w; work=work)
            @test out_vjp ≈ S.matrix' * w atol=1e-10
        end

        @testset "In-place vjp!/jvp! slow path" begin
            prob4 = DCOPFProblem(basic); solve!(prob4)
            @test isnothing(prob4.cache.dz_dd)

            S = calc_sensitivity(DCOPFProblem(basic) |> p -> (solve!(p); p), :lmp, :d)
            w = randn(size(S, 1))
            v = randn(size(S, 2))
            work = zeros(kkt_dims(prob4))

            out_vjp = zeros(size(S, 2))
            vjp!(out_vjp, prob4, :lmp, :d, w; work=work)
            @test out_vjp ≈ S.matrix' * w atol=1e-10

            # Invalidate and test JVP slow path
            invalidate!(prob4.cache)
            prob4.cache.solution = solve!(prob4)
            out_jvp = zeros(size(S, 1))
            jvp!(out_jvp, prob4, :lmp, :d, v; work=work)
            @test out_jvp ≈ S.matrix * v atol=1e-10
        end
    end

    # =================================================================
    # DC OPF — Non-basic network
    # =================================================================
    @testset "DC OPF non-basic" begin
        prob_nb = DCOPFProblem(raw)
        solve!(prob_nb)
        S = calc_sensitivity(prob_nb, :lmp, :d)

        @testset "VJP with Dict (bus ID 10)" begin
            adj = Dict(10 => 1.0)
            result = vjp(prob_nb, :lmp, :d, adj)
            @test sort(collect(keys(result))) == [1, 2, 3, 4, 10]

            # Manual: S'[col_for_10, :] since adj is unit vector at bus 10
            row_idx = S.id_to_row[10]
            for (id, val) in result
                col_idx = S.id_to_col[id]
                @test val ≈ S.matrix[row_idx, col_idx] atol=1e-10
            end
        end

        @testset "JVP with Dict (bus ID 10)" begin
            tang = Dict(10 => 0.1)
            result = jvp(prob_nb, :lmp, :d, tang)
            @test sort(collect(keys(result))) == [1, 2, 3, 4, 10]

            col_idx = S.id_to_col[10]
            for (id, val) in result
                row_idx = S.id_to_row[id]
                @test val ≈ S.matrix[row_idx, col_idx] * 0.1 atol=1e-10
            end
        end
    end

    # =================================================================
    # AC OPF
    # =================================================================
    @testset "AC OPF" begin
        ac_data = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))
        ac_prob = ACOPFProblem(ac_data)
        solve!(ac_prob)

        @testset "VJP matches S' * w" begin
            for (op, param) in [(:lmp, :d), (:vm, :sw), (:pg, :cq),
                                (:qg, :cl), (:va, :fmax), (:qlmp, :qd)]
                S = calc_sensitivity(ac_prob, op, param)
                w = randn(size(S, 1))
                expected = S.matrix' * w
                result = vjp(ac_prob, op, param, w)
                @test result ≈ expected atol=1e-6
            end
        end

        @testset "JVP matches S * v" begin
            for (op, param) in [(:lmp, :d), (:vm, :sw), (:pg, :cq),
                                (:qg, :cl), (:va, :fmax), (:qlmp, :qd)]
                S = calc_sensitivity(ac_prob, op, param)
                v = randn(size(S, 2))
                expected = S.matrix * v
                result = jvp(ac_prob, op, param, v)
                @test result ≈ expected atol=1e-6
            end
        end

        @testset "Transpose identity" begin
            n = ac_prob.network.n
            v = randn(n)
            w = randn(n)
            lhs = dot(w, jvp(ac_prob, :lmp, :d, v))
            rhs = dot(v, vjp(ac_prob, :lmp, :d, w))
            @test lhs ≈ rhs atol=1e-8
        end
    end

    # =================================================================
    # DC Power Flow
    # =================================================================
    @testset "DC PF" begin
        net = DCNetwork(basic)
        n = net.n
        d = zeros(n)
        for (_, load) in basic["load"]
            bus = load["load_bus"]
            d[bus] += load["pd"]
        end
        state = DCPowerFlowState(net, d)

        @testset "VJP matches S' * w" begin
            for (op, param) in [(:va, :d), (:f, :d), (:va, :sw), (:f, :sw),
                                (:va, :b), (:f, :b)]
                S = calc_sensitivity(state, op, param)
                w = randn(size(S, 1))
                expected = S.matrix' * w
                result = vjp(state, op, param, w)
                @test result ≈ expected atol=1e-10
            end
        end

        @testset "JVP matches S * v" begin
            for (op, param) in [(:va, :d), (:f, :d), (:va, :sw), (:f, :sw),
                                (:va, :b), (:f, :b)]
                S = calc_sensitivity(state, op, param)
                v = randn(size(S, 2))
                expected = S.matrix * v
                result = jvp(state, op, param, v)
                @test result ≈ expected atol=1e-10
            end
        end

        @testset "Transpose identity" begin
            v = randn(n)
            w = randn(net.m)
            lhs = dot(w, jvp(state, :f, :d, v))
            rhs = dot(v, vjp(state, :f, :d, w))
            @test lhs ≈ rhs atol=1e-10
        end
    end

    # =================================================================
    # Slow path (no cache) — DC OPF
    # =================================================================
    @testset "DC OPF slow path (no cache)" begin
        # Reference: materialized matrix from a separate problem
        prob_ref = DCOPFProblem(basic); solve!(prob_ref)
        S = calc_sensitivity(prob_ref, :lmp, :d)
        w = randn(size(S, 1))
        v = randn(size(S, 2))

        # Fresh problem — cache is empty
        prob_slow = DCOPFProblem(basic); solve!(prob_slow)
        @test isnothing(prob_slow.cache.dz_dd)

        vjp_result = vjp(prob_slow, :lmp, :d, w)
        @test vjp_result ≈ S.matrix' * w atol=1e-10

        # Invalidate to test JVP slow path too
        invalidate!(prob_slow.cache)
        prob_slow.cache.solution = solve!(prob_slow)
        jvp_result = jvp(prob_slow, :lmp, :d, v)
        @test jvp_result ≈ S.matrix * v atol=1e-10
    end

    # =================================================================
    # Slow path (no cache) — AC OPF
    # =================================================================
    @testset "AC OPF slow path (no cache)" begin
        ac_data2 = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))
        ac_ref = ACOPFProblem(ac_data2); solve!(ac_ref)
        S = calc_sensitivity(ac_ref, :lmp, :d)
        w = randn(size(S, 1))
        v = randn(size(S, 2))

        # Fresh problem
        ac_data3 = PowerModels.parse_file(joinpath(PM_DATA_DIR, "case5.m"))
        ac_slow = ACOPFProblem(ac_data3); solve!(ac_slow)
        @test isnothing(ac_slow.cache.dz_dd)

        vjp_result = vjp(ac_slow, :lmp, :d, w)
        @test vjp_result ≈ S.matrix' * w atol=1e-6

        invalidate!(ac_slow.cache)
        ac_slow.cache.solution = solve!(ac_slow)
        jvp_result = jvp(ac_slow, :lmp, :d, v)
        @test jvp_result ≈ S.matrix * v atol=1e-6
    end

    # =================================================================
    # Error handling
    # =================================================================
    @testset "Error handling" begin
        prob = DCOPFProblem(basic)
        solve!(prob)

        # Invalid operand
        @test_throws ArgumentError vjp(prob, :bad, :d, randn(5))
        @test_throws ArgumentError jvp(prob, :bad, :d, randn(5))

        # Invalid parameter
        @test_throws ArgumentError vjp(prob, :lmp, :bad, randn(5))
        @test_throws ArgumentError jvp(prob, :lmp, :bad, randn(5))

        # Invalid combination
        @test_throws ArgumentError vjp(prob, :im, :d, randn(5))

        # Unknown Dict ID
        @test_throws ArgumentError vjp(prob, :lmp, :d, Dict(9999 => 1.0))
        @test_throws ArgumentError jvp(prob, :lmp, :d, Dict(9999 => 1.0))

        # ACPowerFlowState — not supported, gives helpful error
        pf_data = deepcopy(basic)
        PowerModels.compute_ac_pf!(pf_data)
        ac_state = ACPowerFlowState(pf_data)
        @test_throws ArgumentError vjp(ac_state, :vm, :p, randn(5))
        @test_throws ArgumentError jvp(ac_state, :vm, :p, randn(5))
    end
end
