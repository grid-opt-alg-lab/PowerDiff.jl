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

# FD verification of DC PF susceptance sensitivity (∂θ/∂b, ∂f/∂b). DC PF is
# a direct linear solve (no optimizer), so there is no solver noise — FD
# accuracy is limited only by floating-point cancellation.

@testset "DC PF Susceptance Sensitivity" begin
    net_data = load_test_case("case14.m")
    if isnothing(net_data)
        @test_skip false
    else
        dc_net = DCNetwork(net_data)
        d = calc_demand_vector(net_data)
        state = DCPowerFlowState(dc_net, d)

        dva_db = calc_sensitivity(state, :va, :b)
        df_db = calc_sensitivity(state, :f, :b)

        @testset "Sensitivity metadata" begin
            @test size(dva_db) == (dc_net.n, dc_net.m)
            @test size(df_db) == (dc_net.m, dc_net.m)
            @test dva_db.formulation == :dcpf
            @test dva_db.operand == :va
            @test dva_db.parameter == :b
            @test df_db.formulation == :dcpf
            @test df_db.operand == :f
            @test df_db.parameter == :b
            @test all(isfinite, Matrix(dva_db))
            @test all(isfinite, Matrix(df_db))
        end

        @testset "Finite-difference verification" begin
            # ε=1e-7: tighter perturbation than OPF tests because DC PF is a direct
            # solve — no optimizer noise. Tolerance 1e-4 gives ~1000x margin over
            # the expected FD error O(ε) ≈ 1e-7.
            ε = 1e-7
            W = Diagonal(-dc_net.b .* dc_net.sw)

            # Test a subset of branches for speed
            test_branches = 1:min(5, dc_net.m)
            for e in test_branches
                # Perturb susceptance of branch e
                dc_net_pert = DCNetwork(net_data)
                dc_net_pert.b[e] += ε
                state_pert = DCPowerFlowState(dc_net_pert, d)

                # Finite-difference for va
                fd_dva = (state_pert.va - state.va) / ε
                @test norm(Matrix(dva_db)[:, e] - fd_dva) / max(norm(fd_dva), 1e-10) < 1e-4

                # Finite-difference for flows
                W_pert = Diagonal(-dc_net_pert.b .* dc_net_pert.sw)
                f_base = W * dc_net.A * state.va
                f_pert = W_pert * dc_net_pert.A * state_pert.va
                fd_df = (f_pert - f_base) / ε
                @test norm(Matrix(df_db)[:, e] - fd_df) / max(norm(fd_df), 1e-10) < 1e-4
            end
        end

        @testset "Reference bus row is zero" begin
            ref = dc_net.ref_bus
            @test all(Matrix(dva_db)[ref, :] .== 0.0)
        end
    end
end
