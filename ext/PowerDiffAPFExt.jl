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

module PowerDiffAPFExt

using PowerDiff
import AcceleratedDCPowerFlows as APF
using SparseArrays

# =============================================================================
# AcceleratedDCPowerFlows Interop
# =============================================================================
#
# Direct integration between PowerDiff (PD) and AcceleratedDCPowerFlows
# (APF). Both packages use the B-theta formulation with identical susceptance
# sign conventions and sort elements by original PowerModels key before
# re-indexing, so matrix rows/columns align directly.

# -----------------------------------------------------------------------------
# PTDF convenience
# -----------------------------------------------------------------------------

"""
    materialize_apf_ptdf(Φ::APF.FullPTDF) → Matrix{Float64}

Materialize a dense PTDF matrix from an APF `FullPTDF` object by injecting
identity columns through `compute_flow!`.
"""
function PowerDiff.materialize_apf_ptdf(Φ::APF.FullPTDF)
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
    to_apf_network(net::DCNetwork; allow_fractional=false) → APF.Network

Convert a `DCNetwork` to an `APF.Network`.

APF networks lack generators, costs, and limits, so this is one way.
APF exposes one slack bus, so the converted topology must be a single island.
Bus demand is set to zero (PD separates demand from network topology).
Branch `status` is derived from switching state: `sw[e] > 0.5`, and the
susceptance `b[e]` is passed unscaled.

Because APF models branches as strictly on/off, a fractional `sw[e]` has no
faithful APF representation. By default (`allow_fractional=false`) such switching
states are rejected. Switching sensitivity, however, treats `sw ∈ [0, 1]` as
continuous, so pass `allow_fractional=true` to convert anyway under APF's own
`sw[e] > 0.5` binarization. In either case the single-island requirement is
checked with that same binarization rule, so the slack-bus assumption matches
the topology APF actually builds.

Note: `to_apf_network` sets bus demand to zero because PD separates demand
from topology. For APF workflows that need demand data (e.g., `compute_flow!`),
use `APF.from_power_models(pm_data)` directly instead.
"""
# Single-island check under APF's branch-status rule. APF includes a branch in
# its susceptance matrix iff it is closed (`sw[e] > 0.5`) and carries nonzero
# susceptance, so islands must be counted with that exact rule (not PD's
# `b[e]*sw[e] != 0`) to match the topology APF builds. For binary sw the two
# rules coincide. Returns true iff all buses fall in one connected component.
function _apf_single_island(n::Int, from_bus::Vector{Int}, to_bus::Vector{Int}, active::AbstractVector{Bool})
    parent = collect(1:n)
    function find(i::Int)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end
    @inbounds for e in eachindex(active)
        active[e] || continue
        ri, rj = find(from_bus[e]), find(to_bus[e])
        ri == rj && continue
        ri < rj ? (parent[rj] = ri) : (parent[ri] = rj)
    end
    n <= 1 && return true
    r0 = find(1)
    return all(find(i) == r0 for i in 2:n)
end

function PowerDiff.to_apf_network(net::PowerDiff.DCNetwork; allow_fractional::Bool=false)
    n, m = net.n, net.m

    # APF models branches as strictly on/off and passes b[e] unscaled, so a
    # fractional sw has no faithful APF representation. Reject it unless the caller
    # opts in, in which case we convert under APF's own sw[e] > 0.5 binarization.
    if !allow_fractional
        all(s -> s == 0 || s == 1, net.sw) || throw(ArgumentError(
            "AcceleratedDCPowerFlows conversion requires binary switching states (sw[e] ∈ {0, 1}); " *
            "APF binarizes status via sw[e] > 0.5 and uses b[e] unscaled, so fractional switching has " *
            "no faithful APF representation. Pass allow_fractional=true to convert under the sw > 0.5 rule."
        ))
    end

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

    # Count islands with the rule APF will apply (closed and nonzero susceptance),
    # so the single-slack requirement matches the converted topology even when sw
    # is fractional. For binary sw this matches PD's energized check exactly.
    active = [net.sw[e] > 0.5 && !iszero(net.b[e]) for e in 1:m]
    _apf_single_island(n, from_bus, to_bus, active) || throw(ArgumentError(
        "AcceleratedDCPowerFlows conversion requires one energized island under APF's sw[e] > 0.5 " *
        "binarization; the converted topology would be disconnected"
    ))

    # All buses are active: DCNetwork is built from PM.build_ref() which filters
    # out inactive buses, so every bus in net.n is active by construction.
    buses = [APF.Bus(i, true, 0.0) for i in 1:n]

    branches = Vector{APF.Branch}(undef, m)
    for e in 1:m
        branches[e] = APF.Branch(e, net.sw[e] > 0.5, net.b[e], net.fmax[e], from_bus[e], to_bus[e])
    end

    return APF.Network("PowerDiff", buses, net.ref_bus, branches)
end

# -----------------------------------------------------------------------------
# PTDF / LODF via APF
# -----------------------------------------------------------------------------

"""
    apf_ptdf(net::DCNetwork; kwargs...) → APF.FullPTDF

Build an APF `FullPTDF` from a `DCNetwork`.
Keyword arguments are forwarded to `APF.full_ptdf`.
"""
function PowerDiff.apf_ptdf(net::PowerDiff.DCNetwork; kwargs...)
    return APF.full_ptdf(PowerDiff.to_apf_network(net); kwargs...)
end

"""
    apf_lodf(net::DCNetwork; kwargs...) → APF.FullLODF

Build an APF `FullLODF` from a `DCNetwork`.
Keyword arguments are forwarded to `APF.full_lodf`.
"""
function PowerDiff.apf_lodf(net::PowerDiff.DCNetwork; kwargs...)
    return APF.full_lodf(PowerDiff.to_apf_network(net); kwargs...)
end

# -----------------------------------------------------------------------------
# PTDF Cross-Validation
# -----------------------------------------------------------------------------

"""
    compare_ptdf(state::DCPowerFlowState; atol=1e-8) → (match::Bool, maxerr::Float64)

Cross-validate PD's PTDF against APF's FullPTDF.
Returns a named tuple where `match` is true if all entries agree within `atol`.

Note: This is not cheap — it computes two full PTDF matrices (one via PD's
sensitivity API, one via APF). Intended for validation, not hot-path use.
"""
function PowerDiff.compare_ptdf(state::PowerDiff.DCPowerFlowState; atol::Float64=1e-8)
    pd_ptdf = PowerDiff.ptdf_matrix(state)
    apf_ptdf_mat = PowerDiff.materialize_apf_ptdf(PowerDiff.apf_ptdf(state.net))
    maxerr = maximum(abs, pd_ptdf - apf_ptdf_mat)
    return (match = maxerr < atol, maxerr = maxerr)
end

end # module PowerDiffAPFExt
