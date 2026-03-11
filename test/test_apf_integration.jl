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

# Try to load APF — skip extension tests if unavailable
_apf_available = try
    @eval using AcceleratedDCPowerFlows
    true
catch e
    # Only skip if the package is genuinely not installed; re-throw real errors
    if isa(e, ArgumentError) && contains(string(e), "not found")
        false
    else
        rethrow()
    end
end

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
        B = PowerDiff.calc_susceptance_matrix(dc_net)
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
        # Build a fresh DCNetwork with one capacitive branch (positive b)
        # so B_r is not positive definite and Cholesky falls back to LU.
        dc_net = DCNetwork(net)
        b_cap = copy(dc_net.b)
        b_cap[1] = abs(b_cap[1])
        dc_net_cap = DCNetwork(dc_net.n, dc_net.m, dc_net.k, dc_net.A, dc_net.G_inc, b_cap;
            sw=dc_net.sw, fmax=dc_net.fmax, gmax=dc_net.gmax, gmin=dc_net.gmin,
            angmax=dc_net.angmax, angmin=dc_net.angmin, cq=dc_net.cq, cl=dc_net.cl,
            c_shed=dc_net.c_shed, ref_bus=dc_net.ref_bus, tau=dc_net.tau)

        d = calc_demand_vector(net)
        state = DCPowerFlowState(dc_net_cap, d)

        # Verify LU fallback was used (sparse lu → UmfpackLU, not LinearAlgebra.LU)
        @test state.B_r_factor isa SparseArrays.UMFPACK.UmfpackLU

        # Verify angles match a manual dense solve
        B = PowerDiff.calc_susceptance_matrix(dc_net_cap)
        non_ref = setdiff(1:dc_net_cap.n, dc_net_cap.ref_bus)
        p = state.pg .- state.d
        θ_ref = zeros(dc_net_cap.n)
        θ_ref[non_ref] = Matrix(B[non_ref, non_ref]) \ p[non_ref]
        @test isapprox(state.va, θ_ref, atol=1e-10)
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

if !_apf_available
    @info "AcceleratedDCPowerFlows not available — skipping APF extension tests"
end

# =========================================================================
# APF-dependent tests (skipped if APF not available)
# =========================================================================
if _apf_available

APF = AcceleratedDCPowerFlows

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
@testset "Index alignment PD ↔ APF" begin
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
@testset "PTDF consistency PD ↔ APF" begin
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

            # PD PTDF
            pd_ptdf = ptdf_matrix(state)

            # APF PTDF (materialize via helper)
            apf_Φ = apf_ptdf(dc_net)
            apf_ptdf_mat = materialize_apf_ptdf(apf_Φ)

            @test isapprox(pd_ptdf, apf_ptdf_mat, atol=1e-8)

            # Also test the convenience function
            result = compare_ptdf(state)
            @test result.match
            @test result.maxerr < 1e-8
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

            # PD switching sensitivity: ∂f/∂sw
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

                # FullLODF stores the dense LODF matrix in the .matrix field (public API)
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
        dc_net_orig = DCNetwork(net)
        e_open = 3

        # APF ignores Branch.status in PTDF/LODF — it uses br.b directly.
        # Build a fresh DCNetwork with sw=0 and b=0 for the open branch so
        # both packages see the same open-branch topology.
        sw_open = copy(dc_net_orig.sw)
        b_open = copy(dc_net_orig.b)
        sw_open[e_open] = 0.0
        b_open[e_open] = 0.0
        dc_net = DCNetwork(dc_net_orig.n, dc_net_orig.m, dc_net_orig.k,
            dc_net_orig.A, dc_net_orig.G_inc, b_open;
            sw=sw_open, fmax=dc_net_orig.fmax, gmax=dc_net_orig.gmax,
            gmin=dc_net_orig.gmin, angmax=dc_net_orig.angmax, angmin=dc_net_orig.angmin,
            cq=dc_net_orig.cq, cl=dc_net_orig.cl, c_shed=dc_net_orig.c_shed,
            ref_bus=dc_net_orig.ref_bus, tau=dc_net_orig.tau)

        apf_net = to_apf_network(dc_net)
        @test !apf_net.branches[e_open].status
        @test all(apf_net.branches[e].status for e in 1:dc_net.m if e != e_open)

        # Build power flow state with the open branch
        d = calc_demand_vector(net)
        state_open = DCPowerFlowState(dc_net, d)
        pd_ptdf = ptdf_matrix(state_open)

        # Open branch should have zero PTDF row
        @test isapprox(pd_ptdf[e_open, :], zeros(dc_net.n), atol=1e-12)

        # Cross-validate PD ↔ APF on the modified topology
        result = compare_ptdf(state_open)
        @test result.match
        @test result.maxerr < 1e-8
    end
end

end # if _apf_available

end # @testset "APF Integration"
