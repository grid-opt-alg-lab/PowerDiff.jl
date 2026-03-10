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
# Uses ForwardDiff on polar power flow equations with full Y matrix
# (includes shunts and transformer models).

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
    va0 = angle.(state.v)
    vm0 = abs.(state.v)
    n = state.n

    function pq_polar(x)
        va = x[1:n]
        vm = x[n+1:2n]
        v = vm .* cis.(va)
        S = v .* conj.(Y_dense * v)
        return vcat(real.(S), imag.(S))
    end

    J = ForwardDiff.jacobian(pq_polar, vcat(va0, vm0))

    dp_dva = J[1:n, 1:n]
    dp_dvm = J[1:n, n+1:2n]
    dq_dva = J[n+1:2n, 1:n]
    dq_dvm = J[n+1:2n, n+1:2n]

    if bus_types !== nothing
        for j in 1:n
            if bus_types[j] == 2  # PV bus
                dp_dvm[:, j] .= 0.0
                dq_dvm[:, j] .= 0.0
                dq_dvm[j, j] = 1.0
            elseif bus_types[j] == 3  # Slack bus
                dp_dva[:, j] .= 0.0
                dp_dva[j, j] = 1.0
                dp_dvm[:, j] .= 0.0
                dq_dva[:, j] .= 0.0
                dq_dvm[:, j] .= 0.0
                dq_dvm[j, j] = 1.0
            end
        end
    end

    return (dp_dva=dp_dva, dp_dvm=dp_dvm, dq_dva=dq_dva, dq_dvm=dq_dvm)
end
