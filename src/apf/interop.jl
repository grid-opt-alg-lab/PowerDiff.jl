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
# AcceleratedDCPowerFlows Interop
# =============================================================================
#
# Direct integration between PowerModelsDiff (PMD) and AcceleratedDCPowerFlows
# (APF). Both packages use the B-theta formulation with identical susceptance
# sign conventions and sort elements by original PowerModels key before
# re-indexing, so matrix rows/columns align directly.

# -----------------------------------------------------------------------------
# PTDF convenience
# -----------------------------------------------------------------------------

"""
    ptdf_matrix(state::DCPowerFlowState) → Matrix{Float64}

Return the standard PTDF matrix (`∂f/∂p`) from a DC power flow state.

PMD's `calc_sensitivity(state, :f, :d)` computes `∂f/∂d = -PTDF` because
`p = g - d ⟹ ∂p/∂d = -I`. This function negates to recover the standard
PTDF sign convention: `PTDF = ∂f/∂p`.
"""
ptdf_matrix(state::DCPowerFlowState) = -Matrix(calc_sensitivity(state, :f, :d))

"""
    materialize_apf_ptdf(Φ::APF.FullPTDF) → Matrix{Float64}

Materialize a dense PTDF matrix from an APF `FullPTDF` object by injecting
identity columns through `compute_flow!`.
"""
function materialize_apf_ptdf(Φ::APF.FullPTDF)
    ptdf = zeros(Φ.E, Φ.N)
    ei = zeros(Φ.N)
    for i in 1:Φ.N
        ei[i] = 1.0
        APF.compute_flow!(@view(ptdf[:, i]), ei, Φ)
        ei[i] = 0.0
    end
    return ptdf
end

# -----------------------------------------------------------------------------
# Network Conversion
# -----------------------------------------------------------------------------

"""
    to_apf_network(net::DCNetwork) → APF.Network

Convert a `DCNetwork` to an `APF.Network`.

APF networks lack generators, costs, and limits, so this is one-way.
Bus demand is set to zero (PMD separates demand from network topology).
Branch `status` is derived from switching state: `sw[e] > 0.5`.

Note: `to_apf_network` sets bus demand to zero because PMD separates demand
from topology. For APF workflows that need demand data (e.g., `compute_flow!`),
use `APF.from_power_models(pm_data)` directly instead.
"""
function to_apf_network(net::DCNetwork)
    n, m = net.n, net.m

    # All buses are active: DCNetwork is built from PM.build_ref() which filters
    # out inactive buses, so every bus in net.n is active by construction.
    buses = [APF.Bus(i, true, 0.0) for i in 1:n]

    from_bus = zeros(Int, m)
    to_bus = zeros(Int, m)
    I, J, V = findnz(net.A)
    for k in eachindex(I)
        if V[k] > 0
            from_bus[I[k]] = J[k]
        else
            to_bus[I[k]] = J[k]
        end
    end
    @assert all(>(0), from_bus) && all(>(0), to_bus) "Incidence matrix A must have exactly one +1 and one -1 per row"

    branches = Vector{APF.Branch}(undef, m)
    for e in 1:m
        branches[e] = APF.Branch(e, net.sw[e] > 0.5, net.b[e], net.fmax[e], from_bus[e], to_bus[e])
    end

    return APF.Network("PowerModelsDiff", buses, net.ref_bus, branches)
end

# -----------------------------------------------------------------------------
# PTDF / LODF via APF
# -----------------------------------------------------------------------------

"""
    apf_ptdf(net::DCNetwork; kwargs...) → APF.FullPTDF

Build an APF `FullPTDF` from a `DCNetwork`.
Keyword arguments are forwarded to `APF.full_ptdf`.
"""
function apf_ptdf(net::DCNetwork; kwargs...)
    return APF.full_ptdf(to_apf_network(net); kwargs...)
end

"""
    apf_lodf(net::DCNetwork; kwargs...) → APF.FullLODF

Build an APF `FullLODF` from a `DCNetwork`.
Keyword arguments are forwarded to `APF.full_lodf`.
"""
function apf_lodf(net::DCNetwork; kwargs...)
    return APF.full_lodf(to_apf_network(net); kwargs...)
end

# -----------------------------------------------------------------------------
# PTDF Cross-Validation
# -----------------------------------------------------------------------------

"""
    compare_ptdf(state::DCPowerFlowState; atol=1e-8) → (match::Bool, maxerr::Float64)

Cross-validate PMD's PTDF against APF's FullPTDF.
Returns a named tuple where `match` is true if all entries agree within `atol`.

Note: This is not cheap — it computes two full PTDF matrices (one via PMD's
sensitivity API, one via APF). Intended for validation, not hot-path use.
"""
function compare_ptdf(state::DCPowerFlowState; atol::Float64=1e-8)
    pmd_ptdf = ptdf_matrix(state)
    apf_ptdf_mat = materialize_apf_ptdf(apf_ptdf(state.net))
    maxerr = maximum(abs, pmd_ptdf - apf_ptdf_mat)
    return (match = maxerr < atol, maxerr = maxerr)
end
