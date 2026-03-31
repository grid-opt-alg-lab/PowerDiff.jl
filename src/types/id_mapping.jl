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
        issorted(bus_ids) || throw(ArgumentError("bus_ids must be sorted"))
        issorted(branch_ids) || throw(ArgumentError("branch_ids must be sorted"))
        issorted(gen_ids) || throw(ArgumentError("gen_ids must be sorted"))
        issorted(load_ids) || throw(ArgumentError("load_ids must be sorted"))
        issorted(shunt_ids) || throw(ArgumentError("shunt_ids must be sorted"))
        length(bus_ids) == length(bus_to_idx) || throw(ArgumentError(
            "bus_ids length ($(length(bus_ids))) must match bus_to_idx length ($(length(bus_to_idx)))"))
        length(branch_ids) == length(branch_to_idx) || throw(ArgumentError(
            "branch_ids length ($(length(branch_ids))) must match branch_to_idx length ($(length(branch_to_idx)))"))
        length(gen_ids) == length(gen_to_idx) || throw(ArgumentError(
            "gen_ids length ($(length(gen_ids))) must match gen_to_idx length ($(length(gen_to_idx)))"))
        length(load_ids) == length(load_to_idx) || throw(ArgumentError(
            "load_ids length ($(length(load_ids))) must match load_to_idx length ($(length(load_to_idx)))"))
        length(shunt_ids) == length(shunt_to_idx) || throw(ArgumentError(
            "shunt_ids length ($(length(shunt_ids))) must match shunt_to_idx length ($(length(shunt_to_idx)))"))
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

@inline _sorted_int_keys(tbl::AbstractDict) = sort(parse.(Int, collect(keys(tbl))))

function IDMapping(net::Dict{String,<:Any})
    haskey(net, "bus") || throw(ArgumentError("network missing required key \"bus\""))
    haskey(net, "branch") || throw(ArgumentError("network missing required key \"branch\""))
    haskey(net, "gen") || throw(ArgumentError("network missing required key \"gen\""))
    isempty(net["bus"]) && throw(ArgumentError("Network has no buses"))

    bus_ids = _sorted_int_keys(net["bus"])
    branch_ids = sort([parse(Int, id) for (id, br) in net["branch"] if get(br, "br_status", 1) != 0])
    gen_ids = sort([parse(Int, id) for (id, gen) in net["gen"] if get(gen, "gen_status", 1) != 0])
    if haskey(net, "load")
        load_ids = sort([parse(Int, id) for (id, load) in net["load"] if get(load, "status", 1) != 0])
    else
        load_ids = sort([parse(Int, id) for (id, bus) in net["bus"]
                         if !iszero(get(bus, "pd", 0.0)) || !iszero(get(bus, "qd", 0.0))])
    end
    if haskey(net, "shunt")
        shunt_ids = sort([parse(Int, id) for (id, shunt) in net["shunt"] if get(shunt, "status", 1) != 0])
    else
        shunt_ids = sort([parse(Int, id) for (id, bus) in net["bus"]
                          if !iszero(get(bus, "gs", 0.0)) || !iszero(get(bus, "bs", 0.0))])
    end

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
    _prepare_network_data(net::Dict) → (pm_data, id_map)

Preprocess a PowerModels network dictionary exactly once.
Returns a deepcopy with standardized costs/thermal limits and the IDMapping.
"""
function _prepare_network_data(net::Dict)
    pm_data = deepcopy(net)
    PM.standardize_cost_terms!(pm_data, order=2)
    PM.calc_thermal_limits!(pm_data)
    id_map = IDMapping(pm_data)
    return (pm_data, id_map)
end
