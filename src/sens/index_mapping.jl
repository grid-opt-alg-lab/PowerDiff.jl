# =============================================================================
# Index Mapping Utilities for Sensitivity Results
# =============================================================================
#
# Provides bidirectional index mappings between internal matrix indices
# and external element IDs (bus numbers, branch indices, generator indices).
#
# Following PowerModels' AdmittanceMatrix pattern for consistent indexing.

"""
    _bus_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional bus index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external bus ID
- `id_to_idx::Dict{Int,Int}`: External bus ID → internal row/col index
"""
function _bus_mapping(state::DCPowerFlowState)
    n = state.net.n
    idx_to_id = collect(1:n)  # DC formulation uses 1:n indexing
    id_to_idx = Dict(i => i for i in 1:n)
    return idx_to_id, id_to_idx
end

function _bus_mapping(prob::DCOPFProblem)
    n = prob.network.n
    idx_to_id = collect(1:n)
    id_to_idx = Dict(i => i for i in 1:n)
    return idx_to_id, id_to_idx
end

function _bus_mapping(state::ACPowerFlowState)
    n = state.n
    idx_to_id = collect(1:n)
    id_to_idx = Dict(i => i for i in 1:n)
    return idx_to_id, id_to_idx
end

function _bus_mapping(prob::ACOPFProblem)
    n = prob.network.n
    idx_to_id = collect(1:n)
    id_to_idx = Dict(i => i for i in 1:n)
    return idx_to_id, id_to_idx
end

"""
    _branch_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional branch index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external branch ID
- `id_to_idx::Dict{Int,Int}`: External branch ID → internal row/col index
"""
function _branch_mapping(state::DCPowerFlowState)
    m = state.net.m
    idx_to_id = collect(1:m)
    id_to_idx = Dict(i => i for i in 1:m)
    return idx_to_id, id_to_idx
end

function _branch_mapping(prob::DCOPFProblem)
    m = prob.network.m
    idx_to_id = collect(1:m)
    id_to_idx = Dict(i => i for i in 1:m)
    return idx_to_id, id_to_idx
end

function _branch_mapping(state::ACPowerFlowState)
    m = state.m
    idx_to_id = collect(1:m)
    id_to_idx = Dict(i => i for i in 1:m)
    return idx_to_id, id_to_idx
end

function _branch_mapping(prob::ACOPFProblem)
    m = prob.network.m
    idx_to_id = collect(1:m)
    id_to_idx = Dict(i => i for i in 1:m)
    return idx_to_id, id_to_idx
end

"""
    _gen_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional generator index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external generator ID
- `id_to_idx::Dict{Int,Int}`: External generator ID → internal row/col index
"""
function _gen_mapping(prob::DCOPFProblem)
    k = prob.network.k
    idx_to_id = collect(1:k)
    id_to_idx = Dict(i => i for i in 1:k)
    return idx_to_id, id_to_idx
end

function _gen_mapping(prob::ACOPFProblem)
    k = prob.n_gen
    idx_to_id = collect(1:k)
    id_to_idx = Dict(i => i for i in 1:k)
    return idx_to_id, id_to_idx
end

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
