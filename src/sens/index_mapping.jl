# =============================================================================
# Index Mapping Utilities for Sensitivity Results
# =============================================================================
#
# Provides bidirectional index mappings between internal matrix indices
# and external element IDs (bus numbers, branch indices, generator indices).
#
# Design note: All mappings are currently identity (1:n → 1:n) because
# make_basic_network() guarantees sequential 1-based indexing. The mapping
# infrastructure exists so that Sensitivity{T} exposes row_to_id/col_to_id
# as public API. Supporting non-basic networks in the future would only
# require implementing non-identity versions of _bus_mapping(), _branch_mapping(),
# and _gen_mapping() — no changes to the Sensitivity type or interface.jl needed.
#
# Following PowerModels' AdmittanceMatrix pattern for consistent indexing.
# Uses a module-level cache for identity mappings to avoid repeated allocation.

# Module-level cache: dimension → (idx_to_id, id_to_idx)
const _IDENTITY_MAPPING_CACHE = Dict{Int, Tuple{Vector{Int}, Dict{Int,Int}}}()

"""
    _identity_mapping(n::Int) → (idx_to_id, id_to_idx)

Get or create a cached identity mapping for dimension `n`.
Since `make_basic_network()` produces sequential 1-based indexing,
all mappings are identity (i → i).
"""
function _identity_mapping(n::Int)
    get!(_IDENTITY_MAPPING_CACHE, n) do
        idx_to_id = collect(1:n)
        id_to_idx = Dict(i => i for i in 1:n)
        (idx_to_id, id_to_idx)
    end
end

"""
    _bus_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional bus index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external bus ID
- `id_to_idx::Dict{Int,Int}`: External bus ID → internal row/col index
"""
_bus_mapping(state::DCPowerFlowState) = _identity_mapping(state.net.n)
_bus_mapping(prob::DCOPFProblem) = _identity_mapping(prob.network.n)
_bus_mapping(state::ACPowerFlowState) = _identity_mapping(state.n)
_bus_mapping(prob::ACOPFProblem) = _identity_mapping(prob.network.n)

"""
    _branch_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional branch index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external branch ID
- `id_to_idx::Dict{Int,Int}`: External branch ID → internal row/col index
"""
_branch_mapping(state::DCPowerFlowState) = _identity_mapping(state.net.m)
_branch_mapping(prob::DCOPFProblem) = _identity_mapping(prob.network.m)
_branch_mapping(state::ACPowerFlowState) = _identity_mapping(state.m)
_branch_mapping(prob::ACOPFProblem) = _identity_mapping(prob.network.m)

"""
    _gen_mapping(state) → (idx_to_id, id_to_idx)

Build bidirectional generator index mappings from a state object.

Returns:
- `idx_to_id::Vector{Int}`: Internal row/col index → external generator ID
- `id_to_idx::Dict{Int,Int}`: External generator ID → internal row/col index
"""
_gen_mapping(prob::DCOPFProblem) = _identity_mapping(prob.network.k)
_gen_mapping(prob::ACOPFProblem) = _identity_mapping(prob.n_gen)

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
