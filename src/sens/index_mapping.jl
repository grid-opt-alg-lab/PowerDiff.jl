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
# Index Mapping Utilities for Sensitivity Results
# =============================================================================
#
# Provides bidirectional index mappings between internal matrix indices
# and external element IDs (bus numbers, branch indices, generator indices).
#
# Mappings are derived from the IDMapping stored in each network/state object.
# For basic networks (sequential 1-based IDs), these are identity mappings.
# For non-basic networks, they map original IDs to sequential indices.

"""
    _mapping_from(ids::Vector{Int}) → (idx_to_id, id_to_idx)

Create a bidirectional mapping from a sorted vector of original IDs.
"""
function _mapping_from(ids::Vector{Int})
    idx_to_id = ids  # Sensitivity constructor will copy
    id_to_idx = Dict(id => i for (i, id) in enumerate(ids))
    (idx_to_id, id_to_idx)
end

"""
    _bus_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional bus index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external bus ID
- `id_to_idx::Dict{Int,Int}`: External bus ID → internal row/col index
"""
_bus_mapping(state::DCPowerFlowState) = _mapping_from(state.net.id_map.bus_ids)
_bus_mapping(prob::DCOPFProblem) = _mapping_from(prob.network.id_map.bus_ids)
_bus_mapping(prob::ACOPFProblem) = _mapping_from(prob.network.id_map.bus_ids)

function _bus_mapping(state::ACPowerFlowState)
    if !isnothing(state.net)
        return _mapping_from(state.net.id_map.bus_ids)
    end
    # Fallback for states constructed without a network (e.g. from raw Y matrix)
    ids = collect(1:state.n)
    return (ids, Dict(i => i for i in 1:state.n))
end

"""
    _branch_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional branch index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external branch ID
- `id_to_idx::Dict{Int,Int}`: External branch ID → internal row/col index
"""
_branch_mapping(state::DCPowerFlowState) = _mapping_from(state.net.id_map.branch_ids)
_branch_mapping(prob::DCOPFProblem) = _mapping_from(prob.network.id_map.branch_ids)
_branch_mapping(prob::ACOPFProblem) = _mapping_from(prob.network.id_map.branch_ids)

function _branch_mapping(state::ACPowerFlowState)
    if !isnothing(state.net)
        return _mapping_from(state.net.id_map.branch_ids)
    end
    ids = collect(1:state.m)
    return (ids, Dict(i => i for i in 1:state.m))
end

"""
    _gen_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional generator index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external generator ID
- `id_to_idx::Dict{Int,Int}`: External generator ID → internal row/col index
"""
_gen_mapping(prob::DCOPFProblem) = _mapping_from(prob.network.id_map.gen_ids)
_gen_mapping(prob::ACOPFProblem) = _mapping_from(prob.network.id_map.gen_ids)

# Power flow states don't have generator-level dispatch
function _gen_mapping(::DCPowerFlowState)
    error("DCPowerFlowState does not have generator-level variables")
end

function _gen_mapping(::ACPowerFlowState)
    error("ACPowerFlowState does not have generator-level variables")
end

"""
    _element_mapping(state, element::Symbol) → (idx_to_id, id_to_idx)

Dispatch helper for element-specific index mappings.

# Arguments
- `state`: Power flow state or OPF problem
- `element`: One of `:bus`, `:branch`, or `:gen`

# Returns
Tuple of (idx_to_id::Vector{Int}, id_to_idx::Dict{Int,Int})
"""
function _element_mapping(state, element::Symbol)
    if element === :bus
        return _bus_mapping(state)
    elseif element === :branch
        return _branch_mapping(state)
    elseif element === :gen
        return _gen_mapping(state)
    else
        error("Unknown element type: $element. Expected :bus, :branch, or :gen")
    end
end
