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

const APF = PowerModelsDiff.APF

@testset "APF Integration" begin

# =========================================================================
# Cholesky factorization (independent of APF)
# =========================================================================
@testset "Cholesky factorization in DCPowerFlowState" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        d = calc_demand_vector(net)
        state = DCPowerFlowState(dc_net, d)

        # Verify Cholesky is used for standard inductive networks
        # On sparse matrices, cholesky(Symmetric(...)) returns CHOLMOD.Factor
        @test !(state.B_r_factor isa LU)

        # Verify angles match a manual LU solve
        B = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        non_ref = setdiff(1:dc_net.n, dc_net.ref_bus)

        # Compare against the state's own injection
        p = state.pg .- state.d
        θ_ref = zeros(dc_net.n)
        θ_ref[non_ref] = lu(B[non_ref, non_ref]) \ p[non_ref]
        @test isapprox(state.va, θ_ref, atol=1e-10)
    end
end

# =========================================================================
# Cholesky → LU fallback for capacitive branches
# =========================================================================
@testset "Cholesky → LU fallback for capacitive branch" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)

        # Make one branch capacitive (positive b) so B_r is not positive definite
        dc_net.b[1] = abs(dc_net.b[1])

        d = calc_demand_vector(net)
        state = DCPowerFlowState(dc_net, d)

        # Verify LU fallback was used (sparse lu → UmfpackLU, not LinearAlgebra.LU)
        @test state.B_r_factor isa SparseArrays.UMFPACK.UmfpackLU

        # Verify angles match a manual dense solve
        B = PowerModelsDiff.calc_susceptance_matrix(dc_net)
        non_ref = setdiff(1:dc_net.n, dc_net.ref_bus)
        p = state.pg .- state.d
        θ_ref = zeros(dc_net.n)
        θ_ref[non_ref] = Matrix(B[non_ref, non_ref]) \ p[non_ref]
        @test isapprox(state.va, θ_ref, atol=1e-10)

        # Restore original susceptance
        dc_net.b[1] = -abs(dc_net.b[1])
    end
end

# =========================================================================
# Network conversion
# =========================================================================
@testset "to_apf_network" begin
    net = load_test_case("case14.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        apf_net = to_apf_network(dc_net)

        # Dimensions match
        @test APF.num_buses(apf_net) == dc_net.n
        @test APF.num_branches(apf_net) == dc_net.m

        # Slack bus matches
        @test apf_net.slack_bus_index == dc_net.ref_bus

        # Susceptances match
        for e in 1:dc_net.m
            @test apf_net.branches[e].b ≈ dc_net.b[e]
        end

        # Flow limits match
        for e in 1:dc_net.m
            @test apf_net.branches[e].pmax ≈ dc_net.fmax[e]
        end

        # Incidence matrix matches: verify from/to bus assignments
        A = dc_net.A
        for e in 1:dc_net.m
            br = apf_net.branches[e]
            @test A[e, br.bus_fr] ≈ 1.0
            @test A[e, br.bus_to] ≈ -1.0
        end

        # Branch status from switching
        dc_net.sw .= 1.0
        apf_net2 = to_apf_network(dc_net)
        @test all(br.status for br in apf_net2.branches)

        dc_net.sw[1] = 0.0
        apf_net3 = to_apf_network(dc_net)
        @test !apf_net3.branches[1].status
        @test all(apf_net3.branches[e].status for e in 2:dc_net.m)
        dc_net.sw[1] = 1.0  # restore
    end
end

# =========================================================================
# Index alignment
# =========================================================================
@testset "Index alignment PMD ↔ APF" begin
    net = load_test_case("case5.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        apf_net = to_apf_network(dc_net)

        # Both packages sort by original PM key, so sequential indices align.
        # Verify via incidence matrix: sparse(APF.A) ≈ dc_net.A
        A_apf = sparse(APF.branch_incidence_matrix(apf_net))
        @test A_apf ≈ dc_net.A
    end
end

# =========================================================================
# PTDF consistency
# =========================================================================
@testset "PTDF consistency PMD ↔ APF" begin
    for case in ["case5.m", "case14.m"]
        @testset "$case" begin
            net = load_test_case(case)
            if isnothing(net)
                @test_skip false
                continue
            end
            dc_net = DCNetwork(net)
            d = calc_demand_vector(net)
            state = DCPowerFlowState(dc_net, d)

            # PMD PTDF
            pmd_ptdf = ptdf_matrix(state)

            # APF PTDF (materialize via helper)
            apf_Φ = apf_ptdf(dc_net)
            apf_ptdf_mat = materialize_apf_ptdf(apf_Φ)

            @test isapprox(pmd_ptdf, apf_ptdf_mat, atol=1e-8)

            # Also test the convenience function
            result = compare_ptdf(state)
            @test result.match
            @test result.maxerr < 1e-8
        end
    end
end

# =========================================================================
# Standalone ptdf_matrix test (independent of APF)
# =========================================================================
@testset "ptdf_matrix == -calc_sensitivity(:f, :d)" begin
    for case in ["case5.m", "case14.m"]
        @testset "$case" begin
            net = load_test_case(case)
            if isnothing(net)
                @test_skip false
                continue
            end
            dc_net = DCNetwork(net)
            d = calc_demand_vector(net)
            state = DCPowerFlowState(dc_net, d)

            ptdf = ptdf_matrix(state)
            df_dd = -Matrix(calc_sensitivity(state, :f, :d))
            @test isapprox(ptdf, df_dd, atol=1e-12)
        end
    end
end

# =========================================================================
# LODF ↔ switching sensitivity relationship
# =========================================================================
@testset "LODF ↔ switching sensitivity" begin
    for case in ["case5.m", "case14.m"]
        @testset "$case" begin
            net = load_test_case(case)
            if isnothing(net)
                @test_skip false
                continue
            end
            dc_net = DCNetwork(net)
            d = calc_demand_vector(net)
            state = DCPowerFlowState(dc_net, d)

            # PMD switching sensitivity: ∂f/∂sw
            df_dsw = Matrix(calc_sensitivity(state, :f, :sw))

            # APF LODF
            L = apf_lodf(dc_net)

            # The exact relationship (derived from Sherman-Morrison):
            #   LODF[k, e] = -∂f_k/∂sw_e / ∂f_e/∂sw_e   for k ≠ e
            #   LODF[e, e] = -1                            (by convention)
            #
            # The self-sensitivity ∂f_e/∂sw_e in the denominator naturally
            # captures the Sherman-Morrison correction factor, making this
            # relationship exact (not just first-order).
            for e in 1:dc_net.m
                if abs(df_dsw[e, e]) < 1e-10
                    continue  # skip branches with zero self-sensitivity
                end

                lodf_col = L.matrix[:, e]

                mask = trues(dc_net.m)
                mask[e] = false

                if maximum(abs.(lodf_col[mask])) < 1e-10
                    continue  # trivial column
                end

                predicted = -df_dsw[mask, e] / df_dsw[e, e]
                @test isapprox(lodf_col[mask], predicted, atol=1e-10)
            end
        end
    end
end

# =========================================================================
# Non-basic network through conversion
# =========================================================================
@testset "Non-basic network conversion" begin
    # Use PM_DATA_DIR / parse_file directly (not load_test_case) to keep non-basic IDs
    case_path = joinpath(PM_DATA_DIR, "case5.m")
    if !isfile(case_path)
        @test_skip false
    else
        # case5.m has non-sequential bus IDs when not made basic
        raw = PowerModels.parse_file(case_path)
        dc_net = DCNetwork(raw)  # non-basic network
        apf_net = to_apf_network(dc_net)

        @test APF.num_buses(apf_net) == dc_net.n
        @test APF.num_branches(apf_net) == dc_net.m
        @test apf_net.slack_bus_index == dc_net.ref_bus

        # Susceptances should still match
        for e in 1:dc_net.m
            @test apf_net.branches[e].b ≈ dc_net.b[e]
        end

        # PTDF should still work
        d = calc_demand_vector(raw)
        state = DCPowerFlowState(dc_net, d)
        result = compare_ptdf(state)
        @test result.match
    end
end

# =========================================================================
# Open-branch conversion
# =========================================================================
@testset "Open-branch PTDF/LODF via APF" begin
    net = load_test_case("case14.m")
    if isnothing(net)
        @test_skip false
    else
        dc_net = DCNetwork(net)
        e_open = 3

        # APF ignores Branch.status in PTDF/LODF — it uses br.b directly.
        # Zero both sw and b so both packages see the same open-branch topology.
        b_orig = dc_net.b[e_open]
        dc_net.sw[e_open] = 0.0
        dc_net.b[e_open] = 0.0

        apf_net = to_apf_network(dc_net)
        @test !apf_net.branches[e_open].status
        @test all(apf_net.branches[e].status for e in 1:dc_net.m if e != e_open)

        # Build power flow state with the open branch
        d = calc_demand_vector(net)
        state_open = DCPowerFlowState(dc_net, d)
        pmd_ptdf = ptdf_matrix(state_open)

        # Open branch should have zero PTDF row
        @test isapprox(pmd_ptdf[e_open, :], zeros(dc_net.n), atol=1e-12)

        # Cross-validate PMD ↔ APF on the modified topology
        result = compare_ptdf(state_open)
        @test result.match
        @test result.maxerr < 1e-8

        # Restore originals
        dc_net.sw[e_open] = 1.0
        dc_net.b[e_open] = b_orig
    end
end

end # @testset "APF Integration"
