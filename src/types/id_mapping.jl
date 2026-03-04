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
- `bus_to_idx::Dict{Int,Int}`: Original bus ID → sequential index
- `branch_to_idx::Dict{Int,Int}`: Original branch ID → sequential index
- `gen_to_idx::Dict{Int,Int}`: Original generator ID → sequential index
- `load_to_idx::Dict{Int,Int}`: Original load ID → sequential index
"""
struct IDMapping
    bus_ids::Vector{Int}
    branch_ids::Vector{Int}
    gen_ids::Vector{Int}
    load_ids::Vector{Int}
    bus_to_idx::Dict{Int,Int}
    branch_to_idx::Dict{Int,Int}
    gen_to_idx::Dict{Int,Int}
    load_to_idx::Dict{Int,Int}
end

"""
    IDMapping(ref::Dict)

Construct IDMapping from a `PM.build_ref` reference dictionary.

Extracts sorted keys from `ref[:bus]`, `ref[:branch]`, `ref[:gen]`, `ref[:load]`
and builds inverse dictionaries.
"""
function IDMapping(ref::Dict)
    bus_ids = sort(collect(keys(ref[:bus])))
    branch_ids = sort(collect(keys(ref[:branch])))
    gen_ids = sort(collect(keys(ref[:gen])))
    load_ids = sort(collect(keys(ref[:load])))

    bus_to_idx = Dict(id => i for (i, id) in enumerate(bus_ids))
    branch_to_idx = Dict(id => i for (i, id) in enumerate(branch_ids))
    gen_to_idx = Dict(id => i for (i, id) in enumerate(gen_ids))
    load_to_idx = Dict(id => i for (i, id) in enumerate(load_ids))

    return IDMapping(bus_ids, branch_ids, gen_ids, load_ids,
                     bus_to_idx, branch_to_idx, gen_to_idx, load_to_idx)
end

"""
    IDMapping(n::Int, m::Int, k::Int, n_load::Int)

Create an identity mapping (1:n → 1:n, etc.) for direct/programmatic constructors.
"""
function IDMapping(n::Int, m::Int, k::Int, n_load::Int)
    bus_ids = collect(1:n)
    branch_ids = collect(1:m)
    gen_ids = collect(1:k)
    load_ids = collect(1:n_load)

    bus_to_idx = Dict(i => i for i in 1:n)
    branch_to_idx = Dict(i => i for i in 1:m)
    gen_to_idx = Dict(i => i for i in 1:k)
    load_to_idx = Dict(i => i for i in 1:n_load)

    return IDMapping(bus_ids, branch_ids, gen_ids, load_ids,
                     bus_to_idx, branch_to_idx, gen_to_idx, load_to_idx)
end

function Base.show(io::IO, m::IDMapping)
    print(io, "IDMapping($(length(m.bus_ids)) buses, $(length(m.branch_ids)) branches, ",
          "$(length(m.gen_ids)) gens, $(length(m.load_ids)) loads)")
end
