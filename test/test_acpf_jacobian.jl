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

# FD verification of AC PF Jacobian blocks (∂P/∂θ, ∂P/∂|V|, ∂Q/∂θ, ∂Q/∂|V|).
# These are exact analytical formulas (not computed via implicit function theorem),
# so tighter tolerances are used.
#
# Also cross-checks against PowerModels' calc_basic_jacobian_matrix.

using PowerDiff
using PowerModels
using ForwardDiff
using LinearAlgebra
using SparseArrays
using Test

function _pq_polar_jacobian(Y, va, vm)
    n = length(vm)
    pq_polar(x) = begin
        va_x = x[1:n]
        vm_x = x[n+1:2n]
        v_x = vm_x .* cis.(va_x)
        S_x = v_x .* conj.(Y * v_x)
        return vcat(real.(S_x), imag.(S_x))
    end

    J = ForwardDiff.jacobian(pq_polar, vcat(va, vm))
    return (
        dp_dva = J[1:n, 1:n],
        dp_dvm = J[1:n, n+1:2n],
        dq_dva = J[n+1:2n, 1:n],
        dq_dvm = J[n+1:2n, n+1:2n],
    )
end

@testset "AC PF Jacobian Verification" begin
    pm_path = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")
    file = joinpath(pm_path, "case5.m")

    pm_data = PowerModels.parse_file(file)
    net_data = PowerModels.make_basic_network(pm_data)

    pf_data = deepcopy(net_data)
    PowerModels.compute_ac_pf!(pf_data)

    # Use PowerModels' admittance matrix for consistency with PM's Jacobian
    v_base = PowerModels.calc_basic_bus_voltage(pf_data)
    Y_pm = PowerModels.calc_basic_admittance_matrix(pf_data)
    pf_ref = PowerModels.build_ref(deepcopy(pf_data))[:it][:pm][:nw][0]
    slack = first(keys(pf_ref[:ref_buses]))

    state = ACPowerFlowState(v_base, Y_pm; idx_slack=slack)

    n = state.n
    v = state.v
    Y = state.Y

    # =========================================================================
    # Finite-difference verification
    # =========================================================================

    delta = 1e-7
    fd_tol = 1e-5
    # delta=1e-7 (tighter than 1e-5 used for IFT-based sensitivities) because
    # Jacobian blocks are exact analytical formulas with no solver noise.
    # FD error is O(delta) ≈ 1e-7; tolerance 1e-5 provides a 100x safety margin.

    # Compute P, Q at base operating point
    S_base = v .* conj.(Y * v)
    P_base = real.(S_base)
    Q_base = imag.(S_base)
    va_base = angle.(v)
    vm_base = abs.(v)

    @testset "∂P/∂θ (finite difference)" begin
        J1 = calc_sensitivity(state, :p, :va)
        for k in 1:n
            va_pert = copy(va_base)
            va_pert[k] += delta
            v_pert = vm_base .* cis.(va_pert)
            P_pert = real.(v_pert .* conj.(Y * v_pert))
            fd_col = (P_pert - P_base) / delta
            @test norm(Matrix(J1)[:, k] - fd_col) / max(norm(fd_col), 1e-10) < fd_tol
        end
    end

    @testset "∂P/∂|V| (finite difference)" begin
        J2 = calc_sensitivity(state, :p, :vm)
        for k in 1:n
            vm_pert = copy(vm_base)
            vm_pert[k] += delta
            v_pert = vm_pert .* cis.(va_base)
            P_pert = real.(v_pert .* conj.(Y * v_pert))
            fd_col = (P_pert - P_base) / delta
            @test norm(Matrix(J2)[:, k] - fd_col) / max(norm(fd_col), 1e-10) < fd_tol
        end
    end

    @testset "∂Q/∂θ (finite difference)" begin
        J3 = calc_sensitivity(state, :q, :va)
        for k in 1:n
            va_pert = copy(va_base)
            va_pert[k] += delta
            v_pert = vm_base .* cis.(va_pert)
            Q_pert = imag.(v_pert .* conj.(Y * v_pert))
            fd_col = (Q_pert - Q_base) / delta
            @test norm(Matrix(J3)[:, k] - fd_col) / max(norm(fd_col), 1e-10) < fd_tol
        end
    end

    @testset "∂Q/∂|V| (finite difference)" begin
        J4 = calc_sensitivity(state, :q, :vm)
        for k in 1:n
            vm_pert = copy(vm_base)
            vm_pert[k] += delta
            v_pert = vm_pert .* cis.(va_base)
            Q_pert = imag.(v_pert .* conj.(Y * v_pert))
            fd_col = (Q_pert - Q_base) / delta
            @test norm(Matrix(J4)[:, k] - fd_col) / max(norm(fd_col), 1e-10) < fd_tol
        end
    end

    # =========================================================================
    # Y matrix consistency with PowerModels
    # =========================================================================

    @testset "Y matrix consistency with PowerModels" begin
        Y_pm = PowerModels.calc_basic_admittance_matrix(pf_data)
        @test norm(Matrix(state.Y) - Matrix(Y_pm)) < 1e-12
    end

    # =========================================================================
    # Cross-check with PowerModels Jacobian (bus-type enforcement)
    # =========================================================================

    @testset "Cross-check with PowerModels Jacobian (basic network)" begin
        bus_types = [pf_data["bus"]["$i"]["bus_type"] for i in 1:n]
        jac = calc_power_flow_jacobian(state; bus_types=bus_types)

        pm_jac = PowerModels.calc_basic_jacobian_matrix(pf_data)
        pm_J = Matrix(pm_jac)

        tol = 1e-10
        @test norm(jac.dp_dva - pm_J[1:n, 1:n]) / norm(pm_J[1:n, 1:n]) < tol
        @test norm(jac.dp_dvm - pm_J[1:n, n+1:2n]) / norm(pm_J[1:n, n+1:2n]) < tol
        @test norm(jac.dq_dva - pm_J[n+1:2n, 1:n]) / norm(pm_J[n+1:2n, 1:n]) < tol
        @test norm(jac.dq_dvm - pm_J[n+1:2n, n+1:2n]) / norm(pm_J[n+1:2n, n+1:2n]) < tol
    end

    # =========================================================================
    # Metadata checks
    # =========================================================================

    @testset "Sensitivity metadata" begin
        J1 = calc_sensitivity(state, :p, :va)
        @test J1.formulation == :acpf
        @test J1.operand == :p
        @test J1.parameter == :va
        @test size(J1) == (n, n)
    end

    # =========================================================================
    # Adversarial states: compare closed-form formulas to ForwardDiff
    # =========================================================================

    @testset "Adversarial states vs ForwardDiff" begin
        cases = [
            (
                name = "zero voltage bus",
                Y = Y_pm,
                va = copy(va_base),
                vm = [vm_base[1], 0.0, vm_base[3], vm_base[4], vm_base[5]],
            ),
            (
                name = "tiny voltages and large angles",
                Y = Y_pm,
                va = [3π, -4π / 3, 7π / 5, -9π / 4, 11π / 6],
                vm = [1e-12, 0.37, 1.4, 1e-9, 0.82],
            ),
            (
                name = "disconnected bus",
                Y = let Y_disc = Matrix(Y_pm)
                    Y_disc[5, :] .= 0.0
                    Y_disc[:, 5] .= 0.0
                    sparse(Y_disc)
                end,
                va = [0.0, 0.4, -1.1, 2.2, -0.7],
                vm = [1.0, 0.55, 1e-10, 1.3, 0.9],
            ),
        ]

        for case in cases
            @testset "$(case.name)" begin
                v_case = case.vm .* cis.(case.va)
                state_case = ACPowerFlowState(v_case, case.Y; idx_slack=slack)
                jac = calc_power_flow_jacobian(state_case)
                fd = _pq_polar_jacobian(case.Y, angle.(state_case.v), abs.(state_case.v))

                @test all(isfinite, jac.dp_dva)
                @test all(isfinite, jac.dp_dvm)
                @test all(isfinite, jac.dq_dva)
                @test all(isfinite, jac.dq_dvm)

                @test jac.dp_dva ≈ fd.dp_dva atol=1e-10 rtol=1e-10
                @test jac.dp_dvm ≈ fd.dp_dvm atol=1e-10 rtol=1e-10
                @test jac.dq_dva ≈ fd.dq_dva atol=1e-10 rtol=1e-10
                @test jac.dq_dvm ≈ fd.dq_dvm atol=1e-10 rtol=1e-10
            end
        end
    end
end
