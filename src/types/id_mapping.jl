# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
    IDMapping

Bidirectional mapping between original network element IDs and sequential
1-based indices used for internal computation. Loads and shunts are aggregated
per bus, so only bus, branch, and generator IDs are tracked.
"""
struct IDMapping
    bus_ids::Vector{Int}
    branch_ids::Vector{Int}
    gen_ids::Vector{Int}
    bus_to_idx::Dict{Int,Int}
    branch_to_idx::Dict{Int,Int}
    gen_to_idx::Dict{Int,Int}

    function IDMapping(bus_ids, branch_ids, gen_ids,
                       bus_to_idx, branch_to_idx, gen_to_idx)
        for (ids, mapping, label) in (
            (bus_ids, bus_to_idx, "bus"),
            (branch_ids, branch_to_idx, "branch"),
            (gen_ids, gen_to_idx, "generator"),
        )
            issorted(ids) || throw(ArgumentError("$label IDs must be sorted"))
            length(ids) == length(mapping) || throw(ArgumentError(
                "$label ID count must match mapping size"))
        end
        new(bus_ids, branch_ids, gen_ids, bus_to_idx, branch_to_idx, gen_to_idx)
    end
end

"""
    IDMapping(data::NamedTuple)

Construct an ID mapping from PowerDiff network tables (see `_network_data`).
"""
function IDMapping(data::NamedTuple)
    isempty(data.bus) && throw(ArgumentError("Network has no buses"))
    bus_ids = sort([b.bus_i for b in data.bus])
    branch_ids = sort([br.index for br in data.branch])
    gen_ids = sort([g.index for g in data.gen])
    return IDMapping(
        bus_ids, branch_ids, gen_ids,
        Dict(id => i for (i, id) in enumerate(bus_ids)),
        Dict(id => i for (i, id) in enumerate(branch_ids)),
        Dict(id => i for (i, id) in enumerate(gen_ids)),
    )
end

"""
    IDMapping(n::Int, m::Int, k::Int)

Create identity mappings for direct programmatic constructors.
"""
function IDMapping(n::Int, m::Int, k::Int)
    return IDMapping(
        collect(1:n), collect(1:m), collect(1:k),
        Dict(i => i for i in 1:n), Dict(i => i for i in 1:m), Dict(i => i for i in 1:k),
    )
end

function Base.show(io::IO, mapping::IDMapping)
    print(io, "IDMapping($(length(mapping.bus_ids)) buses, ",
        "$(length(mapping.branch_ids)) branches, $(length(mapping.gen_ids)) gens)")
end
