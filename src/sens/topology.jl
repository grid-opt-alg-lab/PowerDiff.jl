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
# DC Power Flow Switching Sensitivity
# =============================================================================

"""
    calc_sensitivity_switching(state::DCPowerFlowState) → NamedTuple

Compute switching sensitivity for DC power flow (not OPF).

For DC power flow `θ_r = B_r⁻¹ p_r`, the sensitivity of angles w.r.t. switching is:

    ∂θ_r/∂swₑ = -B_r⁻¹ · (∂B_r/∂swₑ) · θ_r

where `∂B_r/∂swₑ = -bₑ · a_{e,r} · a_{e,r}'` is a rank-1 update from the incidence
column of branch `e` restricted to non-reference buses, and `B_r` is the susceptance
matrix with the reference bus row and column deleted.

# Arguments
- `state`: DCPowerFlowState containing the solved power flow

# Returns
NamedTuple with:
- `dva_dsw`: Jacobian ∂va/∂sw (n × m) - voltage angles w.r.t. switching
- `df_dsw`: Jacobian ∂f/∂sw (m × m) - flows w.r.t. switching
"""
function calc_sensitivity_switching(state::DCPowerFlowState)
    net = state.net
    n, m = net.n, net.m
    nr = state.non_ref

    θ_r = state.va[nr]

    # Build all RHS columns at once, then batch-solve
    n_r = length(nr)
    RHS = zeros(n_r, m)
    for e in 1:m
        a_e_r = net.A[e, nr]
        coeff = -net.b[e] * dot(a_e_r, θ_r)
        RHS[:, e] = Vector(coeff * a_e_r)
    end
    X = -(state.B_r_factor \ RHS)

    # Embed into full n×m matrix (reference bus rows stay zero)
    dva_dsw = zeros(n, m)
    dva_dsw[nr, :] = X

    # Flow sensitivity: f = W · A · va where W = Diag(-b ⊙ sw)
    # ∂f/∂swₑ' = W * A * ∂va/∂swₑ' + direct effect on edge e'
    df_dsw = zeros(m, m)

    W = Diagonal(-net.b .* net.sw)
    for e_prime in 1:m
        # Indirect effect: all edges feel the change in va
        df_dsw[:, e_prime] = W * net.A * dva_dsw[:, e_prime]

        # Direct effect: only edge e_prime
        df_dsw[e_prime, e_prime] += -net.b[e_prime] * dot(net.A[e_prime, :], state.va)
    end

    return (dva_dsw=dva_dsw, df_dsw=df_dsw)
end

"""
    calc_sensitivity_susceptance(state::DCPowerFlowState) → NamedTuple

Compute susceptance sensitivity for DC power flow (not OPF).

For DC power flow `θ_r = B_r⁻¹ p_r`, the sensitivity of angles w.r.t. susceptance is:

    ∂θ_r/∂bₑ = -B_r⁻¹ · (∂B_r/∂bₑ) · θ_r

where `∂B_r/∂bₑ = -swₑ · a_{e,r} · a_{e,r}'` is a rank-1 update from the incidence
column of branch `e` restricted to non-reference buses. This mirrors the switching
sensitivity with `bₑ` replaced by `swₑ` in the coefficient.

# Arguments
- `state`: DCPowerFlowState containing the solved power flow

# Returns
NamedTuple with:
- `dva_db`: Jacobian ∂va/∂b (n × m) - voltage angles w.r.t. susceptances
- `df_db`: Jacobian ∂f/∂b (m × m) - flows w.r.t. susceptances
"""
function calc_sensitivity_susceptance(state::DCPowerFlowState)
    net = state.net
    n, m = net.n, net.m
    nr = state.non_ref

    θ_r = state.va[nr]

    # Build all RHS columns at once, then batch-solve
    n_r = length(nr)
    RHS = zeros(n_r, m)
    for e in 1:m
        a_e_r = net.A[e, nr]
        coeff = -net.sw[e] * dot(a_e_r, θ_r)
        RHS[:, e] = Vector(coeff * a_e_r)
    end
    X = -(state.B_r_factor \ RHS)

    # Embed into full n×m matrix (reference bus rows stay zero)
    dva_db = zeros(n, m)
    dva_db[nr, :] = X

    # Flow sensitivity: f = W · A · va where W = Diag(-b ⊙ sw)
    # ∂f/∂bₑ = W * A * ∂va/∂bₑ + direct effect on edge e
    df_db = zeros(m, m)

    W = Diagonal(-net.b .* net.sw)
    for e in 1:m
        # Indirect effect: all edges feel the change in va
        df_db[:, e] = W * net.A * dva_db[:, e]

        # Direct effect: ∂(-bₑ·swₑ)/∂bₑ · (A·va)ₑ = -swₑ · (A·va)ₑ
        df_db[e, e] += -net.sw[e] * dot(net.A[e, :], state.va)
    end

    return (dva_db=dva_db, df_db=df_db)
end

"""
    calc_sensitivity_demand(state::DCPowerFlowState) → NamedTuple

Compute demand sensitivity for DC power flow (not OPF).

For DC power flow `θ_r = B_r⁻¹ p_r`, the sensitivity of angles w.r.t. demand is:

    ∂va/∂d = -B_r⁻¹  (embedded in the non-reference block)

since `p = g - d` and `∂p/∂d = -I`.

# Arguments
- `state`: DCPowerFlowState containing the solved power flow

# Returns
NamedTuple with:
- `dva_dd`: Jacobian ∂va/∂d (n × n) - voltage angles w.r.t. demand
- `df_dd`: Jacobian ∂f/∂d (m × n) - flows w.r.t. demand
"""
function calc_sensitivity_demand(state::DCPowerFlowState)
    net = state.net
    n = net.n
    nr = state.non_ref

    # dθ/dd: solve B_r * X = I for the reduced block, embed in n×n
    # The output is inherently dense (B_r⁻¹), so we use a batched solve.
    dva_dd = zeros(n, n)
    n_r = length(nr)
    dva_dd[nr, nr] = -(state.B_r_factor \ Matrix(1.0I, n_r, n_r))

    # ∂f/∂d = W · A · ∂va/∂d
    W = Diagonal(-net.b .* net.sw)
    df_dd = W * net.A * dva_dd

    return (dva_dd=dva_dd, df_dd=df_dd)
end
