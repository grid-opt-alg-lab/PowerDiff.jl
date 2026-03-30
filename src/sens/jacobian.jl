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
# Power Flow Jacobian Blocks for AC Power Flow
# =============================================================================
#
# Computes the standard 4-block power flow Jacobian:
#   J1 = ∂P/∂θ, J2 = ∂P/∂|V|, J3 = ∂Q/∂θ, J4 = ∂Q/∂|V|
#
# Uses the standard closed-form polar Jacobian formulas on the full Y matrix
# (including shunts and transformer models).

"""
    calc_power_flow_jacobian(state::ACPowerFlowState; bus_types=nothing)

Compute the 4-block power flow Jacobian at the current operating point.

Returns a NamedTuple with:
- `dp_dva`: ∂P/∂θ (n × n) — J1
- `dp_dvm`: ∂P/∂|V| (n × n) — J2
- `dq_dva`: ∂Q/∂θ (n × n) — J3
- `dq_dvm`: ∂Q/∂|V| (n × n) — J4

By default (`bus_types=nothing`), returns raw partial derivatives for ALL buses.

Pass `bus_types::Vector{Int}` (1=PQ, 2=PV, 3=slack) to apply Newton-Raphson
bus-type column modifications matching PowerModels' `calc_basic_jacobian_matrix`:
- **PQ (1)**: Raw derivatives (no modification)
- **PV (2)**: θ columns unchanged; |V| columns zeroed with `∂Q_j/∂|V_j| = 1`
- **Slack (3)**: All columns become unit vectors (`e_j`)
"""
function calc_power_flow_jacobian(state::ACPowerFlowState;
                                  bus_types::Union{Vector{Int}, Nothing}=nothing)
    Y_dense = Matrix(state.Y)
    G = real.(Y_dense)
    B = imag.(Y_dense)
    va0 = angle.(state.v)
    vm0 = abs.(state.v)
    n = state.n
    S = state.v .* conj.(state.Y * state.v)
    P = real.(S)
    Q = imag.(S)

    dp_dva = spzeros(Float64, n, n)
    dp_dvm = spzeros(Float64, n, n)
    dq_dva = spzeros(Float64, n, n)
    dq_dvm = spzeros(Float64, n, n)

    @inbounds for i in 1:n
        Vi = vm0[i]
        θi = va0[i]

        dp_dva[i, i] = -Q[i] - B[i, i] * Vi^2
        dq_dva[i, i] = P[i] - G[i, i] * Vi^2
        dp_dvm_i = 2.0 * G[i, i] * Vi
        dq_dvm_i = -2.0 * B[i, i] * Vi

        for j in 1:n
            i == j && continue
            Vj = vm0[j]
            θij = θi - va0[j]
            cosθ = cos(θij)
            sinθ = sin(θij)
            Gij = G[i, j]
            Bij = B[i, j]
            Vij = Vi * Vj

            dp_dva[i, j] = Vij * (Gij * sinθ - Bij * cosθ)
            dq_dva[i, j] = -Vij * (Gij * cosθ + Bij * sinθ)
            dp_dvm[i, j] = Vi * (Gij * cosθ + Bij * sinθ)
            dq_dvm[i, j] = Vi * (Gij * sinθ - Bij * cosθ)

            dp_dvm_i += Vj * (Gij * cosθ + Bij * sinθ)
            dq_dvm_i += Vj * (Gij * sinθ - Bij * cosθ)
        end

        dp_dvm[i, i] = dp_dvm_i
        dq_dvm[i, i] = dq_dvm_i
    end

    if bus_types !== nothing
        for j in 1:n
            if bus_types[j] == 2  # PV bus
                for i in 1:n
                    dp_dvm[i, j] = 0.0
                    dq_dvm[i, j] = 0.0
                end
                dq_dvm[j, j] = 1.0
            elseif bus_types[j] == 3  # Slack bus
                for i in 1:n
                    dp_dva[i, j] = 0.0
                    dp_dvm[i, j] = 0.0
                    dq_dva[i, j] = 0.0
                    dq_dvm[i, j] = 0.0
                end
                dp_dva[j, j] = 1.0
                dq_dvm[j, j] = 1.0
            end
        end
    end

    return (dp_dva=dp_dva, dp_dvm=dp_dvm, dq_dva=dq_dva, dq_dvm=dq_dvm)
end
