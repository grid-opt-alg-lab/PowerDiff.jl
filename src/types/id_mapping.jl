# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
    IDMapping

Bidirectional mapping between original network element IDs and sequential
1-based indices used for internal computation.
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
        for (ids, mapping, label) in (
            (bus_ids, bus_to_idx, "bus"),
            (branch_ids, branch_to_idx, "branch"),
            (gen_ids, gen_to_idx, "generator"),
            (load_ids, load_to_idx, "load"),
            (shunt_ids, shunt_to_idx, "shunt"),
        )
            issorted(ids) || throw(ArgumentError("$label IDs must be sorted"))
            length(ids) == length(mapping) || throw(ArgumentError(
                "$label ID count must match mapping size"))
        end
        new(bus_ids, branch_ids, gen_ids, load_ids, shunt_ids,
            bus_to_idx, branch_to_idx, gen_to_idx, load_to_idx, shunt_to_idx)
    end
end

"""
    IDMapping(data::ParsedCase)

Construct an ID mapping from normalized typed network data.
"""
function IDMapping(data::ParsedCase)
    isempty(data.bus) && throw(ArgumentError("Network has no buses"))
    bus_ids = sort(getfield.(data.bus, :bus_i))
    branch_ids = sort(getfield.(data.branch, :index))
    gen_ids = sort(getfield.(data.gen, :index))
    load_ids = sort(getfield.(data.load, :index))
    shunt_ids = sort(getfield.(data.shunt, :index))
    return IDMapping(
        bus_ids, branch_ids, gen_ids, load_ids, shunt_ids,
        Dict(id => i for (i, id) in enumerate(bus_ids)),
        Dict(id => i for (i, id) in enumerate(branch_ids)),
        Dict(id => i for (i, id) in enumerate(gen_ids)),
        Dict(id => i for (i, id) in enumerate(load_ids)),
        Dict(id => i for (i, id) in enumerate(shunt_ids)),
    )
end

"""
    IDMapping(n::Int, m::Int, k::Int, n_load::Int; n_shunt::Int=0)

Create identity mappings for direct programmatic constructors.
"""
function IDMapping(n::Int, m::Int, k::Int, n_load::Int; n_shunt::Int=0)
    return IDMapping(
        collect(1:n), collect(1:m), collect(1:k), collect(1:n_load), collect(1:n_shunt),
        Dict(i => i for i in 1:n), Dict(i => i for i in 1:m),
        Dict(i => i for i in 1:k), Dict(i => i for i in 1:n_load),
        Dict(i => i for i in 1:n_shunt),
    )
end

function Base.show(io::IO, mapping::IDMapping)
    print(io, "IDMapping($(length(mapping.bus_ids)) buses, ",
        "$(length(mapping.branch_ids)) branches, $(length(mapping.gen_ids)) gens, ",
        "$(length(mapping.load_ids)) loads, $(length(mapping.shunt_ids)) shunts)")
end
