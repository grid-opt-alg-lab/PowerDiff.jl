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

# Finite-difference verification for all AC OPF sensitivities:
# Parameters: :d, :qd, :cq, :cl, :fmax (+ existing :sw)
# Operands: :va, :vm, :pg, :qg, :lmp

using PowerDiff
using PowerModels
using LinearAlgebra
using Test

"""Check FD agreement for dual sensitivities using the calc function end-to-end."""
function _check_fd_dual(calc_fn, key, sol_base, sol_pert, prob_base, prob_pert, sens, col_idx, ε)
    val_base = calc_fn(sol_base, prob_base)
    val_pert = calc_fn(sol_pert, prob_pert)
    fd = (val_pert - val_base) / ε
    analytical = Matrix(sens[key])[:, col_idx]
    # Dual variables can have smaller magnitudes than primal operands,
    # so use a looser skip threshold (1e-4 vs 1e-6)
    if norm(fd) > 1e-4
        rel_err = norm(analytical - fd) / norm(fd)
        @test rel_err < 1e-2
    end
end

@testset "AC OPF All Sensitivities" begin
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")
    pm_data = PowerModels.parse_file(file)
    pm_data = PowerModels.make_basic_network(pm_data)

    operands = [:va, :vm, :pg, :qg, :lmp, :qlmp]
    n, m, k = 5, 7, 5  # case5 dimensions

    # Expected row sizes for each operand
    op_sizes = Dict(:va => n, :vm => n, :pg => k, :qg => k, :lmp => n, :qlmp => n)
    # Expected col sizes for each parameter
    param_sizes = Dict(:sw => m, :d => n, :qd => n, :cq => k, :cl => k, :fmax => m)

    # =========================================================================
    # Coverage: all 36 combinations produce correct sizes and finite values
    # =========================================================================
    @testset "All 36 combinations — sizes & finiteness" begin
        prob = ACOPFProblem(pm_data; silent=true)

        for param in [:sw, :d, :qd, :cq, :cl, :fmax]
            for op in operands
                @testset "$op w.r.t. $param" begin
                    S = calc_sensitivity(prob, op, param)
                    @test S isa Sensitivity
                    @test S.formulation == :acopf
                    @test S.operand == op
                    @test S.parameter == param
                    @test size(S) == (op_sizes[op], param_sizes[param])
                    @test all(isfinite, Matrix(S))
                end
            end
        end
    end

    # =========================================================================
    # LMP operand for switching (was previously invalid)
    # =========================================================================
    @testset "LMP w.r.t. switching" begin
        prob = ACOPFProblem(pm_data; silent=true)
        dlmp_dsw = calc_sensitivity(prob, :lmp, :sw)
        @test size(dlmp_dsw) == (n, m)
        @test all(isfinite, Matrix(dlmp_dsw))
    end

    # =========================================================================
    # Finite-difference verification: active demand (:d)
    # =========================================================================
    @testset "FD verification: demand (:d)" begin
        prob = ACOPFProblem(pm_data; silent=true)
        sol_base = solve!(prob)

        # Compute all sensitivities
        sens = Dict(op => calc_sensitivity(prob, op, :d) for op in operands)

        ε = 1e-5
        # Find buses with significant load (sorted for determinism)
        load_buses = Int[]
        for (lid, load) in pm_data["load"]
            bus = load["load_bus"]
            if load["pd"] > 0.1 && !(bus in load_buses)
                push!(load_buses, bus)
            end
        end
        sort!(load_buses)

        for bus_idx in load_buses[1:min(2, length(load_buses))]
            pm_pert = deepcopy(pm_data)
            # Perturb all loads at this bus
            for (lid, load) in pm_pert["load"]
                if load["load_bus"] == bus_idx
                    load["pd"] += ε
                end
            end

            prob_pert = ACOPFProblem(pm_pert; silent=true)
            sol_pert = solve!(prob_pert)

            for op in [:va, :vm, :pg, :qg]
                fd = (getfield(sol_pert, op) - getfield(sol_base, op)) / ε
                analytical = Matrix(sens[op])[:, bus_idx]
                if norm(fd) > 1e-6
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 1e-2
                end
            end

            _check_fd_dual(calc_lmp, :lmp, sol_base, sol_pert, prob, prob_pert, sens, bus_idx, ε)
            _check_fd_dual(calc_qlmp, :qlmp, sol_base, sol_pert, prob, prob_pert, sens, bus_idx, ε)
        end
    end

    # =========================================================================
    # Finite-difference verification: reactive demand (:qd)
    # =========================================================================
    @testset "FD verification: reactive demand (:qd)" begin
        prob = ACOPFProblem(pm_data; silent=true)
        sol_base = solve!(prob)

        sens = Dict(op => calc_sensitivity(prob, op, :qd) for op in operands)

        ε = 1e-5
        # Find buses with reactive load
        load_buses = Int[]
        for (lid, load) in pm_data["load"]
            bus = load["load_bus"]
            if abs(load["qd"]) > 0.001 && !(bus in load_buses)
                push!(load_buses, bus)
            end
        end

        for bus_idx in load_buses[1:min(2, length(load_buses))]
            pm_pert = deepcopy(pm_data)
            for (lid, load) in pm_pert["load"]
                if load["load_bus"] == bus_idx
                    load["qd"] += ε
                end
            end

            prob_pert = ACOPFProblem(pm_pert; silent=true)
            sol_pert = solve!(prob_pert)

            for op in [:va, :vm, :pg, :qg]
                fd = (getfield(sol_pert, op) - getfield(sol_base, op)) / ε
                analytical = Matrix(sens[op])[:, bus_idx]
                if norm(fd) > 1e-6
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 1e-2
                end
            end

            _check_fd_dual(calc_lmp, :lmp, sol_base, sol_pert, prob, prob_pert, sens, bus_idx, ε)
            _check_fd_dual(calc_qlmp, :qlmp, sol_base, sol_pert, prob, prob_pert, sens, bus_idx, ε)
        end
    end

    # =========================================================================
    # Finite-difference verification: quadratic cost (:cq)
    # =========================================================================
    @testset "FD verification: quadratic cost (:cq)" begin
        prob = ACOPFProblem(pm_data; silent=true)
        sol_base = solve!(prob)

        sens = Dict(op => calc_sensitivity(prob, op, :cq) for op in operands)

        ε = 1e-5
        for gen_idx in 1:min(2, k)
            pm_pert = deepcopy(pm_data)
            pm_pert["gen"][string(gen_idx)]["cost"][1] += ε

            prob_pert = ACOPFProblem(pm_pert; silent=true)
            sol_pert = solve!(prob_pert)

            for op in [:va, :vm, :pg, :qg]
                fd = (getfield(sol_pert, op) - getfield(sol_base, op)) / ε
                analytical = Matrix(sens[op])[:, gen_idx]
                if norm(fd) > 1e-6
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 1e-2
                end
            end

            _check_fd_dual(calc_lmp, :lmp, sol_base, sol_pert, prob, prob_pert, sens, gen_idx, ε)
            _check_fd_dual(calc_qlmp, :qlmp, sol_base, sol_pert, prob, prob_pert, sens, gen_idx, ε)
        end
    end

    # =========================================================================
    # Finite-difference verification: linear cost (:cl)
    # =========================================================================
    @testset "FD verification: linear cost (:cl)" begin
        prob = ACOPFProblem(pm_data; silent=true)
        sol_base = solve!(prob)

        sens = Dict(op => calc_sensitivity(prob, op, :cl) for op in operands)

        ε = 1e-5
        for gen_idx in 1:min(2, k)
            pm_pert = deepcopy(pm_data)
            pm_pert["gen"][string(gen_idx)]["cost"][2] += ε

            prob_pert = ACOPFProblem(pm_pert; silent=true)
            sol_pert = solve!(prob_pert)

            for op in [:va, :vm, :pg, :qg]
                fd = (getfield(sol_pert, op) - getfield(sol_base, op)) / ε
                analytical = Matrix(sens[op])[:, gen_idx]
                if norm(fd) > 1e-6
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 1e-2
                end
            end

            _check_fd_dual(calc_lmp, :lmp, sol_base, sol_pert, prob, prob_pert, sens, gen_idx, ε)
            _check_fd_dual(calc_qlmp, :qlmp, sol_base, sol_pert, prob, prob_pert, sens, gen_idx, ε)
        end
    end

    # =========================================================================
    # Finite-difference verification: flow limits (:fmax)
    # =========================================================================
    @testset "FD verification: flow limits (:fmax)" begin
        prob = ACOPFProblem(pm_data; silent=true)
        sol_base = solve!(prob)

        sens = Dict(op => calc_sensitivity(prob, op, :fmax) for op in operands)

        ε = 1e-5
        for branch_idx in 1:min(2, m)
            pm_pert = deepcopy(pm_data)
            pm_pert["branch"][string(branch_idx)]["rate_a"] += ε

            prob_pert = ACOPFProblem(pm_pert; silent=true)
            sol_pert = solve!(prob_pert)

            for op in [:va, :vm, :pg, :qg]
                fd = (getfield(sol_pert, op) - getfield(sol_base, op)) / ε
                analytical = Matrix(sens[op])[:, branch_idx]
                if norm(fd) > 1e-6
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 1e-2
                end
            end

            _check_fd_dual(calc_lmp, :lmp, sol_base, sol_pert, prob, prob_pert, sens, branch_idx, ε)
            _check_fd_dual(calc_qlmp, :qlmp, sol_base, sol_pert, prob, prob_pert, sens, branch_idx, ε)
        end
    end

    # =========================================================================
    # Cache reuse: all parameters share KKT factorization
    # =========================================================================
    @testset "Cache reuse across parameters" begin
        prob = ACOPFProblem(pm_data; silent=true)

        # First query triggers solve + factorization
        calc_sensitivity(prob, :vm, :d)
        @test !isnothing(prob.cache.solution)
        @test !isnothing(prob.cache.kkt_factor)
        @test !isnothing(prob.cache.dz_dd)

        # Second parameter reuses factorization
        calc_sensitivity(prob, :pg, :cq)
        @test !isnothing(prob.cache.dz_dcq)

        # Different operand for same parameter reuses dz_dd
        calc_sensitivity(prob, :lmp, :d)
        # No new cache field needed — still uses dz_dd

        # All parameters
        calc_sensitivity(prob, :va, :sw)
        calc_sensitivity(prob, :qg, :qd)
        calc_sensitivity(prob, :vm, :cl)
        calc_sensitivity(prob, :pg, :fmax)
        @test !isnothing(prob.cache.dz_dsw)
        @test !isnothing(prob.cache.dz_dqd)
        @test !isnothing(prob.cache.dz_dcl)
        @test !isnothing(prob.cache.dz_dfmax)
    end

    # =========================================================================
    # Invalidation clears all cached data
    # =========================================================================
    @testset "Invalidation clears cache" begin
        prob = ACOPFProblem(pm_data; silent=true)
        calc_sensitivity(prob, :vm, :d)
        calc_sensitivity(prob, :pg, :sw)

        PowerDiff.invalidate!(prob.cache)

        @test isnothing(prob.cache.solution)
        @test isnothing(prob.cache.kkt_factor)
        @test isnothing(prob.cache.dz_dsw)
        @test isnothing(prob.cache.dz_dd)
        @test isnothing(prob.cache.dz_dqd)
        @test isnothing(prob.cache.dz_dcq)
        @test isnothing(prob.cache.dz_dcl)
        @test isnothing(prob.cache.dz_dfmax)
        @test isnothing(prob.cache.kkt_constants)
    end
end
