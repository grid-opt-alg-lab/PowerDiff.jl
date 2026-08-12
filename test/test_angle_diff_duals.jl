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

# Verifies phase angle difference duals (gamma_lb/gamma_ub) are near-zero when
# angle limits are loose, nonzero when tight, and correctly reflected in KKT
# residuals and FD-verified sensitivities. Tests both upper-limit binding
# (forward flow) and lower-limit binding (reverse flow) scenarios.

using PowerDiff
using LinearAlgebra
using SparseArrays
using Test

import PowerDiff: kkt, kkt_indices, flatten_variables

@testset "Phase Angle Difference Duals" begin

    # 3-bus network: 1→2, 1→3, 2→3
    # Gen at bus 1 (cheap) and bus 2 (expensive), load distributed
    n, m, k = 3, 3, 2
    A = sparse([
        1.0  -1.0   0.0;   # Branch 1: 1→2
        1.0   0.0  -1.0;   # Branch 2: 1→3 (angle-constrained in tight tests)
        0.0   1.0  -1.0    # Branch 3: 2→3
    ])
    G_inc = sparse([
        1.0  0.0;   # Gen 1 at bus 1 (cheap)
        0.0  1.0;   # Gen 2 at bus 2 (expensive)
        0.0  0.0    # No gen at bus 3
    ])
    # All branches have susceptance magnitude 10 p.u. (b = Im(1/z) < 0 for inductive lines)
    b = [-10.0, -10.0, -10.0]

    # Nonzero demand at all buses, small enough to avoid shedding
    d = [0.05, 0.05, 0.5]

    @testset "Loose angle limits (regression)" begin
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=fill(π, m), angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Gamma duals should be ~zero with loose limits
        @test maximum(abs.(sol.gamma_lb)) < 1e-4
        @test maximum(abs.(sol.gamma_ub)) < 1e-4

        # KKT residual: gamma-specific blocks should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d)
        idx = kkt_indices(net)
        @test norm(K[idx.gamma_lb]) < 1e-4
        @test norm(K[idx.gamma_ub]) < 1e-4
        @test norm(K[idx.nu_bal]) < 1e-4
        @test norm(K) < 1e-4  # full KKT residual

        # Sensitivities should be finite
        for (op, param) in [(:va, :d), (:pg, :d), (:f, :d), (:lmp, :d),
                            (:va, :sw), (:pg, :sw)]
            S = calc_sensitivity(prob, op, param)
            @test all(isfinite, Matrix(S))
        end
    end

    @testset "Tight angle limits — binding duals" begin
        # angmax = 0.025 rad on branch 2 (1→3): small enough to bind under given
        # demand pattern (0.5 MW at bus 3) but large enough for feasibility
        angmax_tight = [π, 0.025, π]
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=angmax_tight, angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Branch 2 angle difference should be at or near the limit
        Aθ = net.A * sol.va
        @test Aθ[2] <= angmax_tight[2] + 1e-4

        # No shedding (sufficient gen capacity)
        @test maximum(abs.(sol.psh)) < 1e-6

        # Gamma_ub on branch 2 should be nonzero (binding upper angle limit)
        @test abs(sol.gamma_ub[2]) > 1e-2

        # Gamma duals should be non-negative (standard KKT convention)
        @test all(sol.gamma_lb .>= -1e-6)
        @test all(sol.gamma_ub .>= -1e-6)

        # Gamma-related KKT blocks should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d)
        idx = kkt_indices(net)
        @test norm(K[idx.gamma_lb]) < 1e-3
        @test norm(K[idx.gamma_ub]) < 1e-3
        @test norm(K[idx.nu_bal]) < 1e-4
        @test norm(K) < 1e-3  # full KKT residual

        # All inequality duals should be non-negative (standard KKT convention)
        @test all(sol.lam_ub .>= -1e-6)
        @test all(sol.lam_lb .>= -1e-6)
        @test all(sol.rho_ub .>= -1e-6)
        @test all(sol.rho_lb .>= -1e-6)
        @test all(sol.mu_lb .>= -1e-6)
        @test all(sol.mu_ub .>= -1e-6)

        # FD verification for switching sensitivity with binding angle limits
        dpg_dsw = calc_sensitivity(prob, :pg, :sw)
        df_dsw = calc_sensitivity(prob, :f, :sw)
        @test all(isfinite, Matrix(dpg_dsw))
        @test all(isfinite, Matrix(df_dsw))

        # The matrix-free :sw path must carry the same gated angle difference terms
        # as the materialized Jacobian. `prob_cold` is never passed to
        # calc_sensitivity, so prob.cache.dz_dsw stays empty and vjp/jvp take the
        # inline `_dc_topo_vjp!`/`_dc_topo_jvp!` route rather than the cached
        # matrix multiply. With gamma_ub[2] binding above, dropping those terms
        # makes the two paths disagree.
        prob_cold = DCOPFProblem(net, d)
        solve!(prob_cold)
        v_sw = randn(m)
        w_pg = randn(k)
        w_f = randn(m)
        @test jvp(prob_cold, :pg, :sw, v_sw) ≈ Matrix(dpg_dsw) * v_sw atol=1e-8
        @test jvp(prob_cold, :f, :sw, v_sw) ≈ Matrix(df_dsw) * v_sw atol=1e-8
        @test vjp(prob_cold, :pg, :sw, w_pg) ≈ Matrix(dpg_dsw)' * w_pg atol=1e-8
        @test vjp(prob_cold, :f, :sw, w_f) ≈ Matrix(df_dsw)' * w_f atol=1e-8

        ε = 1e-5
        for e in 1:m
            sw_pert = copy(net.sw)
            sw_pert[e] -= ε

            net.sw .= sw_pert
            prob_pert = DCOPFProblem(net, d)
            sol_pert = solve!(prob_pert)
            net.sw .= ones(m)  # restore

            fd_pg = (sol.pg - sol_pert.pg) / ε
            fd_f  = (sol.f  - sol_pert.f)  / ε

            if norm(fd_pg) > 1e-6
                rel_err = norm(Matrix(dpg_dsw)[:, e] - fd_pg) / norm(fd_pg)
                @test rel_err < 0.05
            end
            if norm(fd_f) > 1e-6
                rel_err = norm(Matrix(df_dsw)[:, e] - fd_f) / norm(fd_f)
                @test rel_err < 0.05
            end
        end
    end

    @testset "Tight lower angle limits — binding angmin" begin
        # Reverse the flow direction: load at bus 1, gen at bus 2 and bus 3
        # This makes A*θ negative on some branches, approaching angmin
        G_inc_rev = sparse([
            0.0  0.0;   # No gen at bus 1 (load)
            1.0  0.0;   # Gen 1 at bus 2 (cheap)
            0.0  1.0    # Gen 2 at bus 3 (expensive)
        ])
        d_rev = [0.5, 0.05, 0.05]

        # Tight angmin on branch 1 (1→2): power flows 2→1, so A*θ < 0
        angmin_tight = [-0.025, -π, -π]
        net = DCNetwork(n, m, k, A, G_inc_rev, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=fill(π, m), angmin=angmin_tight,
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d_rev)
        sol = solve!(prob)

        # Branch 1 angle difference should be at or near the lower limit
        Aθ = net.A * sol.va
        @test Aθ[1] >= angmin_tight[1] - 1e-4

        # Gamma_lb on branch 1 should be nonzero (binding lower angle limit)
        @test abs(sol.gamma_lb[1]) > 1e-2

        # Gamma duals should be non-negative (standard KKT convention)
        @test all(sol.gamma_lb .>= -1e-6)
        @test all(sol.gamma_ub .>= -1e-6)

        # Full KKT residual should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d_rev)
        idx = kkt_indices(net)
        @test norm(K[idx.gamma_lb]) < 1e-3
        @test norm(K[idx.gamma_ub]) < 1e-3
        @test norm(K) < 1e-3

        # FD verification for demand sensitivity
        dg_dd = calc_sensitivity(prob, :pg, :d)
        dva_dd = calc_sensitivity(prob, :va, :d)
        df_dd = calc_sensitivity(prob, :f, :d)
        dlmp_dd = calc_sensitivity(prob, :lmp, :d)

        @test all(isfinite, Matrix(dg_dd))
        @test all(isfinite, Matrix(dva_dd))
        @test all(isfinite, Matrix(df_dd))
        @test all(isfinite, Matrix(dlmp_dd))

        delta = 1e-5
        for bus in 1:n
            d_pert = copy(d_rev)
            d_pert[bus] += delta

            prob_pert = DCOPFProblem(net, d_pert)
            sol_pert = solve!(prob_pert)

            for (name, S, base, pert) in [
                    ("∂g/∂d", dg_dd, sol.pg, sol_pert.pg),
                    ("∂θ/∂d", dva_dd, sol.va, sol_pert.va),
                    ("∂f/∂d", df_dd, sol.f, sol_pert.f),
                    ("∂lmp/∂d", dlmp_dd, sol.nu_bal, sol_pert.nu_bal)]
                fd = (pert - base) / delta
                analytical = Matrix(S)[:, bus]
                if norm(fd) > 1e-8
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 0.01  # 1% tolerance
                end
            end
        end
    end

    @testset "FD verification with binding angle limits" begin
        angmax_tight = [π, 0.025, π]
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=angmax_tight, angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol_base = solve!(prob)

        # Analytical sensitivities
        dg_dd = calc_sensitivity(prob, :pg, :d)
        dva_dd = calc_sensitivity(prob, :va, :d)
        df_dd = calc_sensitivity(prob, :f, :d)
        dlmp_dd = calc_sensitivity(prob, :lmp, :d)

        @test all(isfinite, Matrix(dg_dd))
        @test all(isfinite, Matrix(dva_dd))
        @test all(isfinite, Matrix(df_dd))
        @test all(isfinite, Matrix(dlmp_dd))

        # Finite-difference: perturb demand at each bus
        delta = 1e-5
        for bus in 1:n
            d_pert = copy(d)
            d_pert[bus] += delta

            prob_pert = DCOPFProblem(net, d_pert)
            sol_pert = solve!(prob_pert)

            for (name, S, base, pert) in [
                    ("∂g/∂d", dg_dd, sol_base.pg, sol_pert.pg),
                    ("∂θ/∂d", dva_dd, sol_base.va, sol_pert.va),
                    ("∂f/∂d", df_dd, sol_base.f, sol_pert.f),
                    ("∂lmp/∂d", dlmp_dd, sol_base.nu_bal, sol_pert.nu_bal)]
                fd = (pert - base) / delta
                analytical = Matrix(S)[:, bus]
                if norm(fd) > 1e-8
                    rel_err = norm(analytical - fd) / norm(fd)
                    @test rel_err < 0.01  # 1% tolerance
                end
            end
        end
    end

    @testset "Participation factors still sum to 1" begin
        angmax_tight = [π, 0.025, π]
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=angmax_tight, angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        solve!(prob)

        dg_dd = calc_sensitivity(prob, :pg, :d)
        dpsh_dd = calc_sensitivity(prob, :psh, :d)

        for j in 1:n
            total = sum(Matrix(dg_dd)[:, j]) + sum(Matrix(dpsh_dd)[:, j])
            @test abs(total - 1.0) < 1e-4
        end
    end

    @testset "LMP decomposition with binding angle limits" begin
        angmax_tight = [π, 0.025, π]
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=angmax_tight, angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # Verify energy + congestion ≈ lmp
        lmps = calc_lmp(sol, net)
        energy = calc_energy_component(sol, net)
        congestion = calc_congestion_component(sol, net)
        @test isapprox(lmps, energy .+ congestion, atol=1e-6)

        # With binding angle limit, congestion component should be nonzero
        @test norm(congestion) > 1e-4
    end

    @testset "Simultaneous binding angle and flow limits" begin
        net = DCNetwork(n, m, k, A, G_inc, b;
            fmax=[100.0, 100.0, 0.1],  # tight on branch 3
            gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=[π, 0.025, π],       # tight on branch 2
            angmin=fill(-π, m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)

        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # KKT residual should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d)
        @test norm(K) < 1e-3

        # Both gamma and lambda duals should have nonzero entries
        @test any(abs.(sol.gamma_ub) .> 1e-4)
        @test any(abs.(sol.lam_ub) .> 1e-4) || any(abs.(sol.lam_lb) .> 1e-4)

        # LMP decomposition identity
        lmps = calc_lmp(sol, net)
        energy = calc_energy_component(sol, net)
        congestion = calc_congestion_component(sol, net)
        @test isapprox(lmps, energy .+ congestion, atol=1e-6)

        # Sensitivities should be finite
        dpg_dd = calc_sensitivity(prob, :pg, :d)
        @test all(isfinite, Matrix(dpg_dd))
    end

    @testset "case5 with tight angle limits" begin
        net_dict = load_test_case("case5.m")

        dc_net = DCNetwork(net_dict)
        d_case = calc_demand_vector(net_dict)

        # Tighten angle limits on a few branches
        dc_net.angmax[1] = 0.05
        dc_net.angmax[3] = 0.05

        prob = DCOPFProblem(dc_net, d_case)
        sol = solve!(prob)

        # At least one gamma_ub should be nonzero
        @test any(abs.(sol.gamma_ub) .> 1e-4)

        # Gamma duals should be non-negative
        @test all(sol.gamma_ub .>= -1e-6)
        @test all(sol.gamma_lb .>= -1e-6)

        # KKT gamma blocks should be near zero
        z = flatten_variables(sol, prob)
        K = kkt(z, prob, d_case)
        idx = kkt_indices(dc_net)
        # Ipopt's interior-point method gives slightly looser complementarity
        # than conic solvers on QPs with tight angle limits
        @test norm(K[idx.gamma_lb]) < 5e-3
        @test norm(K[idx.gamma_ub]) < 5e-3
        @test norm(K) < 5e-3  # full KKT residual

        # FD verification for demand sensitivity
        dg_dd = calc_sensitivity(prob, :pg, :d)
        @test all(isfinite, Matrix(dg_dd))

        bus_idx = findfirst(d_case .> 0)
        delta = 1e-5
        d_pert = copy(d_case)
        d_pert[bus_idx] += delta

        prob_pert = DCOPFProblem(dc_net, d_pert)
        sol_pert = solve!(prob_pert)

        dg_fd = (sol_pert.pg - sol.pg) / delta
        if norm(dg_fd) > 1e-8
            rel_err = norm(Matrix(dg_dd)[:, bus_idx] - dg_fd) / norm(dg_fd)
            @test rel_err < 0.05
        end
    end

    @testset "Congestion component gates angle duals by sw" begin
        # Regression (PR #55): calc_congestion_component must gate the angle
        # difference dual term by sw, matching the KKT theta-stationarity term
        # A' * Diag(sw) * (gamma_ub - gamma_lb). That term used to be ungated,
        # which gave a wrong energy/congestion split on any branch with sw != 1
        # and a nonzero angle dual. With sw == 1 everywhere the two forms are
        # identical, so this only shows up on partially switched branches.
        net = DCNetwork(n, m, k, A, G_inc, b; sw=[1.0, 0.5, 1.0],
            fmax=fill(100.0, m), gmax=[5.0, 5.0], gmin=[0.0, 0.0],
            angmax=[Float64(pi), 0.025, Float64(pi)], angmin=fill(-Float64(pi), m),
            cl=[10.0, 50.0], cq=[1.0, 1.0], ref_bus=1, tau=0.01)
        prob = DCOPFProblem(net, d)
        sol = solve!(prob)

        # The fractionally switched branch (2) carries an active angle dual.
        @test abs(sol.gamma_ub[2] - sol.gamma_lb[2]) > 1e-3

        cong = calc_congestion_component(sol, net)
        w = -net.b
        non_ref = PowerDiff._non_reference_buses(net)
        At = net.A'

        # Gated RHS: matches the implementation and the documented stationarity.
        rhs_gated = At * Diagonal(w .* net.sw) * (sol.lam_ub - sol.lam_lb) +
                    At * Diagonal(net.sw) * (sol.gamma_ub - sol.gamma_lb)
        expected = zeros(net.n)
        expected[non_ref] = sol.B_r_factor \ rhs_gated[non_ref]
        @test isapprox(cong, expected; atol=1e-8)

        # Ungated RHS (the pre-fix behavior) disagrees, so the gate is load bearing.
        rhs_ungated = At * Diagonal(w .* net.sw) * (sol.lam_ub - sol.lam_lb) +
                      At * (sol.gamma_ub - sol.gamma_lb)
        ungated = zeros(net.n)
        ungated[non_ref] = sol.B_r_factor \ rhs_ungated[non_ref]
        @test !isapprox(cong, ungated; atol=1e-3)
    end

    @testset "Zero-susceptance angle gating" begin
        A_gate = sparse([
            1.0 -1.0  0.0
            0.0  1.0 -1.0
            1.0  0.0 -1.0
        ])
        G_gate = sparse(Matrix(1.0I, 3, 3))
        d_gate = [0.1, 0.1, 0.1]

        function make_gate_network(b3; exclude_raw_angle=false)
            angle_min = exclude_raw_angle ? [-Float64(pi), -Float64(pi), 0.1] : fill(-Float64(pi), 3)
            angle_max = exclude_raw_angle ? [Float64(pi), Float64(pi), 0.2] : fill(Float64(pi), 3)
            return DCNetwork(3, 3, 3, A_gate, G_gate, [-10.0, -10.0, b3];
                sw=[1.0, 0.8, 0.6], fmax=fill(10.0, 3),
                gmax=fill(10.0, 3), gmin=zeros(3),
                angmin=angle_min, angmax=angle_max,
                cq=ones(3), cl=[10.0, 20.0, 30.0], ref_bus=1, tau=1e-3)
        end

        function with_test_gamma(sol, e)
            gamma_lb = copy(sol.gamma_lb)
            gamma_ub = copy(sol.gamma_ub)
            gamma_lb[e] = 1.25
            gamma_ub[e] = 3.5
            return DCOPFSolution(
                copy(sol.va), copy(sol.pg), copy(sol.f), copy(sol.psh),
                copy(sol.nu_bal), copy(sol.nu_flow),
                copy(sol.lam_ub), copy(sol.lam_lb),
                copy(sol.rho_ub), copy(sol.rho_lb),
                copy(sol.mu_lb), copy(sol.mu_ub),
                gamma_lb, gamma_ub, copy(sol.eta_ref), sol.objective, sol.B_r_factor,
            )
        end

        function finite_difference_kkt_column(net, z, field, e, h)
            values = getfield(net, field)
            base = values[e]
            try
                values[e] = base + h
                plus = kkt(z, net, d_gate)
                values[e] = base - h
                minus = kkt(z, net, d_gate)
                return (plus - minus) / (2h)
            finally
                values[e] = base
                reference_buses(net)  # refresh the topology cache after restoration
            end
        end

        # A zero-b bridge splits the buses into separate energized islands. Its
        # raw angle interval excludes zero, so the old sw-only gate made this
        # otherwise independent two-island problem infeasible.
        A_bridge = sparse(reshape([1.0, -1.0], 1, 2))
        net_bridge = DCNetwork(2, 1, 2, A_bridge, sparse(Matrix(1.0I, 2, 2)), [0.0];
            sw=[1.0], fmax=[10.0], gmax=[10.0, 10.0], gmin=zeros(2),
            angmin=[0.1], angmax=[0.2], cq=ones(2), cl=[10.0, 20.0],
            ref_bus=1, tau=1e-3)
        @test reference_buses(net_bridge) == [1, 2]
        prob_bridge = DCOPFProblem(net_bridge, [0.1, 0.1])
        @test length(prob_bridge.cons.ref) == 2
        sol_bridge = solve!(prob_bridge)
        @test (net_bridge.A * sol_bridge.va)[1] < net_bridge.angmin[1]
        @test iszero(sol_bridge.gamma_lb[1])
        @test iszero(sol_bridge.gamma_ub[1])

        # The zero-b branch is redundant for connectivity, but its raw angle
        # interval deliberately excludes the solved angle. It must impose no model
        # constraint because b[3] * sw[3] == 0.
        net_zero = make_gate_network(0.0; exclude_raw_angle=true)
        @test PowerDiff._angle_difference_gates(net_zero) == [1.0, 0.8, 0.0]
        @test reference_buses(net_zero) == [1]
        prob_zero = DCOPFProblem(net_zero, d_gate)
        sol_zero = solve!(prob_zero)
        @test (net_zero.A * sol_zero.va)[3] < net_zero.angmin[3]
        @test iszero(sol_zero.gamma_lb[3])
        @test iszero(sol_zero.gamma_ub[3])

        z_zero = flatten_variables(sol_zero, prob_zero)
        idx_zero = kkt_indices(prob_zero)
        @test norm(kkt(z_zero, prob_zero, d_gate), Inf) < 1e-5

        # Nonzero test multipliers make every gamma path load bearing without
        # relying on an arbitrary solver dual for the constant 0 <= 0 constraints.
        sol_zero_gamma = with_test_gamma(sol_zero, 3)
        z_zero_gamma = flatten_variables(sol_zero_gamma, prob_zero)
        @test kkt(z_zero_gamma, prob_zero, d_gate) ≈ kkt(z_zero, prob_zero, d_gate) atol=1e-12

        Jz_zero = Matrix(PowerDiff.calc_kkt_jacobian(prob_zero; sol=sol_zero_gamma))
        Jz_zero_fd = ForwardDiff.jacobian(z -> kkt(z, prob_zero, d_gate), z_zero_gamma)
        @test Jz_zero ≈ Jz_zero_fd atol=1e-10 rtol=1e-10
        @test all(iszero, Jz_zero[:, idx_zero.gamma_lb[3]])
        @test all(iszero, Jz_zero[:, idx_zero.gamma_ub[3]])

        Jsw_zero = Matrix(PowerDiff.calc_kkt_jacobian_switching(prob_zero, sol_zero_gamma))
        Jsw_zero_col = PowerDiff.calc_kkt_jacobian_switching_column(prob_zero, sol_zero_gamma, 3)
        Jsw_zero_fd = finite_difference_kkt_column(net_zero, z_zero_gamma, :sw, 3, 1e-6)
        @test all(iszero, Jsw_zero[:, 3])
        @test Jsw_zero_col == Jsw_zero[:, 3]
        @test Jsw_zero[:, 3] ≈ Jsw_zero_fd atol=1e-10

        Jb_zero = Matrix(PowerDiff.calc_kkt_jacobian_susceptance(prob_zero, sol_zero_gamma))
        Jb_zero_plain = Matrix(PowerDiff.calc_kkt_jacobian_susceptance(prob_zero, sol_zero))
        @test Jb_zero ≈ Jb_zero_plain atol=1e-12
        @test PowerDiff.calc_kkt_jacobian_susceptance_column(prob_zero, sol_zero_gamma, 3) ≈ Jb_zero[:, 3] atol=1e-12

        # A synthetic gamma on the zero-b branch must not leak into congestion.
        @test calc_congestion_component(sol_zero_gamma, net_zero) ≈
              calc_congestion_component(sol_zero, net_zero) atol=1e-12

        # On either side of zero, sw_eff == sw and is locally constant in b.
        # Verify both analytic parameter Jacobians without crossing the nonsmooth
        # b == 0 boundary.
        for b3 in (-1e-3, 1e-3)
            net_side = make_gate_network(b3)
            prob_side = DCOPFProblem(net_side, d_gate)
            sol_side = solve!(prob_side)
            sol_side_gamma = with_test_gamma(sol_side, 3)
            z_side_gamma = flatten_variables(sol_side_gamma, prob_side)

            Jsw = Matrix(PowerDiff.calc_kkt_jacobian_switching(prob_side, sol_side_gamma))
            Jsw_plain = Matrix(PowerDiff.calc_kkt_jacobian_switching(prob_side, sol_side))
            Jsw_col = PowerDiff.calc_kkt_jacobian_switching_column(prob_side, sol_side_gamma, 3)
            Jsw_fd = finite_difference_kkt_column(net_side, z_side_gamma, :sw, 3, 1e-6)
            @test Jsw_col ≈ Jsw[:, 3] atol=1e-12
            @test Jsw[:, 3] ≈ Jsw_fd atol=1e-8 rtol=1e-7
            @test norm(Jsw[:, 3] - Jsw_plain[:, 3]) > 1.0

            Jb = Matrix(PowerDiff.calc_kkt_jacobian_susceptance(prob_side, sol_side_gamma))
            Jb_plain = Matrix(PowerDiff.calc_kkt_jacobian_susceptance(prob_side, sol_side))
            Jb_col = PowerDiff.calc_kkt_jacobian_susceptance_column(prob_side, sol_side_gamma, 3)
            Jb_fd = finite_difference_kkt_column(net_side, z_side_gamma, :b, 3, 1e-5)
            @test Jb_col ≈ Jb[:, 3] atol=1e-12
            @test Jb[:, 3] ≈ Jb_fd atol=1e-8 rtol=1e-7
            @test Jb ≈ Jb_plain atol=1e-12
        end
    end
end
