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
# IDMapping: Bidirectional index mapping between original and sequential IDs
# =============================================================================
#
# Translates between original PowerModels element IDs (which may be arbitrary
# integers like [1,2,3,4,10]) and sequential 1-based indices used internally.

"""
    IDMapping

Bidirectional mapping between original PowerModels element IDs and sequential
1-based indices used for internal computation.

Constructed from a `PM.build_ref` dictionary or as an identity mapping for
direct/programmatic constructors.

# Fields
- `bus_ids::Vector{Int}`: Sorted original bus IDs; index position = sequential index
- `branch_ids::Vector{Int}`: Sorted original branch IDs
- `gen_ids::Vector{Int}`: Sorted original generator IDs
- `load_ids::Vector{Int}`: Sorted original load IDs
- `shunt_ids::Vector{Int}`: Sorted original shunt IDs
- `bus_to_idx::Dict{Int,Int}`: Original bus ID → sequential index
- `branch_to_idx::Dict{Int,Int}`: Original branch ID → sequential index
- `gen_to_idx::Dict{Int,Int}`: Original generator ID → sequential index
- `load_to_idx::Dict{Int,Int}`: Original load ID → sequential index
- `shunt_to_idx::Dict{Int,Int}`: Original shunt ID → sequential index
"""
struct IDMapping
    bus_ids::Vector{Int}
    branch_ids::Vector{Int}
    gen_ids::Vector{Int}
    load_ids::Vector{Int}
    shunt_ids::Vector{Int}
    bus_to_idx::Dict{Int,Int}
    branch_to_idx::Dict{Int,Int}
    gen_to_idx::Dict{Int,Int}
    load_to_idx::Dict{Int,Int}
    shunt_to_idx::Dict{Int,Int}

    function IDMapping(bus_ids, branch_ids, gen_ids, load_ids, shunt_ids,
                       bus_to_idx, branch_to_idx, gen_to_idx, load_to_idx, shunt_to_idx)
        @assert issorted(bus_ids) "bus_ids must be sorted"
        @assert issorted(branch_ids) "branch_ids must be sorted"
        @assert issorted(gen_ids) "gen_ids must be sorted"
        @assert issorted(load_ids) "load_ids must be sorted"
        @assert issorted(shunt_ids) "shunt_ids must be sorted"
        @assert length(bus_ids) == length(bus_to_idx)
        @assert length(branch_ids) == length(branch_to_idx)
        @assert length(gen_ids) == length(gen_to_idx)
        @assert length(load_ids) == length(load_to_idx)
        @assert length(shunt_ids) == length(shunt_to_idx)
        new(bus_ids, branch_ids, gen_ids, load_ids, shunt_ids,
            bus_to_idx, branch_to_idx, gen_to_idx, load_to_idx, shunt_to_idx)
    end
end

"""
    IDMapping(ref::Dict)

Construct IDMapping from a `PM.build_ref` reference dictionary.

Extracts sorted keys from `ref[:bus]`, `ref[:branch]`, `ref[:gen]`, `ref[:load]`,
and `ref[:shunt]` (if present) and builds inverse dictionaries.
"""
function IDMapping(ref::Dict)
    for key in (:bus, :branch, :gen, :load)
        haskey(ref, key) || throw(ArgumentError("ref missing required key :$key"))
    end
    isempty(keys(ref[:bus])) && throw(ArgumentError("Network has no buses"))

    bus_ids = sort(collect(keys(ref[:bus])))
    branch_ids = sort(collect(keys(ref[:branch])))
    gen_ids = sort(collect(keys(ref[:gen])))
    load_ids = sort(collect(keys(ref[:load])))
    shunt_ids = sort(collect(keys(get(ref, :shunt, Dict()))))

    bus_to_idx = Dict(id => i for (i, id) in enumerate(bus_ids))
    branch_to_idx = Dict(id => i for (i, id) in enumerate(branch_ids))
    gen_to_idx = Dict(id => i for (i, id) in enumerate(gen_ids))
    load_to_idx = Dict(id => i for (i, id) in enumerate(load_ids))
    shunt_to_idx = Dict(id => i for (i, id) in enumerate(shunt_ids))

    return IDMapping(bus_ids, branch_ids, gen_ids, load_ids, shunt_ids,
                     bus_to_idx, branch_to_idx, gen_to_idx, load_to_idx, shunt_to_idx)
end

"""
    IDMapping(n::Int, m::Int, k::Int, n_load::Int; n_shunt::Int=0)

Create an identity mapping (1:n → 1:n, etc.) for direct/programmatic constructors.
"""
function IDMapping(n::Int, m::Int, k::Int, n_load::Int; n_shunt::Int=0)
    bus_ids = collect(1:n)
    branch_ids = collect(1:m)
    gen_ids = collect(1:k)
    load_ids = collect(1:n_load)
    shunt_ids = collect(1:n_shunt)

    bus_to_idx = Dict(i => i for i in 1:n)
    branch_to_idx = Dict(i => i for i in 1:m)
    gen_to_idx = Dict(i => i for i in 1:k)
    load_to_idx = Dict(i => i for i in 1:n_load)
    shunt_to_idx = Dict(i => i for i in 1:n_shunt)

    return IDMapping(bus_ids, branch_ids, gen_ids, load_ids, shunt_ids,
                     bus_to_idx, branch_to_idx, gen_to_idx, load_to_idx, shunt_to_idx)
end

function Base.show(io::IO, m::IDMapping)
    print(io, "IDMapping($(length(m.bus_ids)) buses, $(length(m.branch_ids)) branches, ",
          "$(length(m.gen_ids)) gens, $(length(m.load_ids)) loads, ",
          "$(length(m.shunt_ids)) shunts)")
end

# =============================================================================
# Network Data Preparation Helper
# =============================================================================

"""
    _prepare_network_data(net::Dict) → (pm_data, ref, id_map)

Preprocess a PowerModels network dictionary exactly once.
Returns a deepcopy with standardized costs/thermal limits, the build_ref result,
and the IDMapping.
"""
function _prepare_network_data(net::Dict)
    pm_data = deepcopy(net)
    PM.standardize_cost_terms!(pm_data, order=2)
    PM.calc_thermal_limits!(pm_data)
    ref = PM.build_ref(pm_data)[:it][:pm][:nw][0]
    id_map = IDMapping(ref)
    return (pm_data, ref, id_map)
end
