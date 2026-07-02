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

@testset "DCPowerFlowState factorization" begin
    @testset "Cholesky factorization in DCPowerFlowState" begin
        net = load_test_case("case5.m")
        if isnothing(net)
            @test_skip false
        else
            dc_net = DCNetwork(net)
            d = calc_demand_vector(net)
            state = DCPowerFlowState(dc_net, d)

            @test !(state.B_r_factor isa LU)

            B = PowerDiff.calc_susceptance_matrix(dc_net)
            non_ref = setdiff(1:dc_net.n, dc_net.ref_bus)

            p = state.pg .- state.d
            theta_ref = zeros(dc_net.n)
            theta_ref[non_ref] = lu(B[non_ref, non_ref]) \ p[non_ref]
            @test isapprox(state.va, theta_ref, atol=1e-10)
        end
    end

    @testset "Cholesky to LU fallback for capacitive branch" begin
        net = load_test_case("case5.m")
        if isnothing(net)
            @test_skip false
        else
            dc_net = DCNetwork(net)
            b_cap = copy(dc_net.b)
            b_cap[1] = abs(b_cap[1])
            dc_net_cap = DCNetwork(dc_net.n, dc_net.m, dc_net.k, dc_net.A, dc_net.G_inc, b_cap;
                sw=dc_net.sw, fmax=dc_net.fmax, gmax=dc_net.gmax, gmin=dc_net.gmin,
                angmax=dc_net.angmax, angmin=dc_net.angmin, cq=dc_net.cq, cl=dc_net.cl,
                c_shed=dc_net.c_shed, ref_bus=dc_net.ref_bus, tau=dc_net.tau)

            d = calc_demand_vector(net)
            state = DCPowerFlowState(dc_net_cap, d)

            @test state.B_r_factor isa SparseArrays.UMFPACK.UmfpackLU

            B = PowerDiff.calc_susceptance_matrix(dc_net_cap)
            non_ref = setdiff(1:dc_net_cap.n, dc_net_cap.ref_bus)
            p = state.pg .- state.d
            theta_ref = zeros(dc_net_cap.n)
            theta_ref[non_ref] = Matrix(B[non_ref, non_ref]) \ p[non_ref]
            @test isapprox(state.va, theta_ref, atol=1e-10)
        end
    end

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
end
