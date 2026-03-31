using LazyArtifacts

const _PD_PRINTABLE_ASCII = 256

is_end(c::Char) = isspace(c) || c in "=;[]%,{}"
const _PD_ENDS = ntuple(i -> is_end(Char(i)), _PD_PRINTABLE_ASCII)

const _PD_NULL_VIEW = SubString("", 1, 0)

struct WordedString
    s::SubString{String}
    len::Int
end

macro iter_to_ntuple(n, iter_expr, types)
    iter_sym = gensym("iter")
    state_sym = gensym("state")
    x_syms = [gensym("x") for _ in 1:n]

    body = Expr[]
    push!(body, :($iter_sym = $(esc(iter_expr))))
    push!(body, :($state_sym = iter_ws($iter_sym, 1)))
    length(types.args) != n && error("types provided to @iter_to_ntuple had length $(length(types.args)) instead of $n")
    for i in 1:n
        push!(body, :($(x_syms[i]) = parse($(esc(types.args[i])), $state_sym[1])))
        if i < n
            push!(body, :($state_sym = $state_sym[2] == 0 ? (_PD_NULL_VIEW, 0) : iter_ws($iter_sym, $state_sym[2])))
        end
    end
    push!(body, Expr(:tuple, x_syms...))
    return Expr(:block, body...)
end

@inline @views function iter_ws(ws::WordedString, start::Int)
    start > ws.len && return (_PD_NULL_VIEW, 0)
    left = start
    while left <= ws.len && isspace(ws.s[left])
        left += 1
    end
    (left > ws.len || ws.s[left] == '%') && return (_PD_NULL_VIEW, 0)
    right = left
    should_end = c -> _PD_ENDS[c]
    while right <= ws.len && !should_end(Int8(ws.s[right]))
        right += 1
    end
    return ws.s[left:right-1], right
end

struct ParsedBus
    bus_i::Int
    bus_type::Int
    pd::Float64
    qd::Float64
    gs::Float64
    bs::Float64
    area::Int
    vm::Float64
    va::Float64
    base_kv::Float64
    zone::Int
    vmax::Float64
    vmin::Float64
end

struct ParsedGen
    index::Int
    gen_bus::Int
    pg::Float64
    qg::Float64
    qmax::Float64
    qmin::Float64
    vg::Float64
    mbase::Float64
    gen_status::Int
    pmax::Float64
    pmin::Float64
    cost::NTuple{3,Float64}
end

struct ParsedBranch
    index::Int
    f_bus::Int
    t_bus::Int
    br_r::Float64
    br_x::Float64
    br_b::Float64
    rate_a::Float64
    rate_b::Float64
    rate_c::Float64
    tap::Float64
    shift::Float64
    br_status::Int
    angmin::Float64
    angmax::Float64
end

struct ParsedLoad
    index::Int
    load_bus::Int
    pd::Float64
    qd::Float64
    status::Int
end

struct ParsedShunt
    index::Int
    shunt_bus::Int
    gs::Float64
    bs::Float64
    status::Int
end

struct ParsedCase
    name::String
    source_version::String
    baseMVA::Float64
    bus::Vector{ParsedBus}
    gen::Vector{ParsedGen}
    branch::Vector{ParsedBranch}
    load::Vector{ParsedLoad}
    shunt::Vector{ParsedShunt}
end


"""
    get_path(library::Symbol)

Resolve an artifact-backed library path owned by PowerDiff.
"""
function get_path(library::Symbol)
    library == :pglib && return joinpath(artifact"PGLib_opf", "pglib-opf-23.07")
    error("unsupported library $(library)")
end


"""
    parse_file(io::Union{IO,String}; library=nothing, import_all=false, validate=true, filetype="json")

Parse a network file.

MATPOWER files are parsed directly into `ParsedCase` to avoid the intermediate
PowerModels dictionary conversion path. PSS/E RAW and JSON files continue to use
PowerModels' parsers and return dictionaries.
"""
function parse_file(io::Union{IO,String}; library=nothing, import_all=false, validate=true, filetype="json")
    resolved = io isa String ? _resolve_case_path(io, library) : io
    local resolved_type = filetype
    if resolved isa String
        resolved_type = lowercase(last(split(resolved, '.')))
    end

    if resolved_type == "m"
        return parse_matpower(resolved; validate)
    elseif resolved_type == "raw"
        return PM.parse_psse(resolved; import_all, validate)
    elseif resolved_type == "json"
        return PM.parse_json(resolved; validate)
    else
        error("unrecognized filetype: .$resolved_type")
    end
end


function parse_matpower(io::IO; validate=true)::ParsedCase
    data_string = read(io, String)
    parsed = _parse_matpower_typed(data_string)
    validate && (parsed = _normalize_parsed_case(parsed))
    validate && _validate_parsed_case(parsed)
    return parsed
end

function parse_matpower(file::String; library=nothing, validate=true)::ParsedCase
    resolved = _resolve_case_path(file, library)
    isfile(resolved) || error("invalid file $(resolved) for library $(library)")
    open(resolved) do io
        parse_matpower(io; validate)
    end
end

function parse_matpower_struct(file::String; library=nothing, validate=true)::ParsedCase
    return parse_matpower(file; library, validate)
end


function _resolve_case_path(path::AbstractString, library)
    if library === nothing
        return String(path)
    end
    return joinpath(get_path(library), path)
end


function _validate_parsed_case(data::ParsedCase)
    isempty(data.bus) && error("no bus table found in matpower file.  The file seems to be missing \"mpc.bus = [...];\"")
    isempty(data.gen) && error("no gen table found in matpower file.  The file seems to be missing \"mpc.gen = [...];\"")
    isempty(data.branch) && error("no branch table found in matpower file.  The file seems to be missing \"mpc.branch = [...];\"")

    bus_ids = Set(b.bus_i for b in data.bus)
    for gen in data.gen
        gen.gen_bus in bus_ids || error("generator $(gen.index) references missing bus $(gen.gen_bus)")
    end
    for branch in data.branch
        branch.f_bus in bus_ids || error("branch $(branch.index) references missing from bus $(branch.f_bus)")
        branch.t_bus in bus_ids || error("branch $(branch.index) references missing to bus $(branch.t_bus)")
    end
    return data
end


function _normalize_parsed_case(data::ParsedCase)::ParsedCase
    pm_data = _parsedcase_to_pm_data(data)
    PM.correct_network_data!(pm_data)
    return _parsedcase_from_pm_data(pm_data)
end


function _parsedcase_to_pm_data(data::ParsedCase)::Dict{String,Any}
    pm_data = Dict{String,Any}(
        "name" => data.name,
        "source_type" => "matpower",
        "source_version" => data.source_version,
        "baseMVA" => data.baseMVA,
        "per_unit" => true,
        "bus" => Dict{String,Any}(),
        "gen" => Dict{String,Any}(),
        "branch" => Dict{String,Any}(),
        "dcline" => Dict{String,Any}(),
        "load" => Dict{String,Any}(),
        "shunt" => Dict{String,Any}(),
        "storage" => Dict{String,Any}(),
        "switch" => Dict{String,Any}(),
    )

    for bus in data.bus
        pm_data["bus"][string(bus.bus_i)] = Dict{String,Any}(
            "index" => bus.bus_i,
            "bus_i" => bus.bus_i,
            "bus_type" => bus.bus_type,
            "pd" => bus.pd,
            "qd" => bus.qd,
            "gs" => bus.gs,
            "bs" => bus.bs,
            "area" => bus.area,
            "vm" => bus.vm,
            "va" => bus.va,
            "base_kv" => bus.base_kv,
            "zone" => bus.zone,
            "vmax" => bus.vmax,
            "vmin" => bus.vmin,
            "source_id" => ["bus", bus.bus_i],
        )
    end

    for gen in data.gen
        pm_data["gen"][string(gen.index)] = Dict{String,Any}(
            "index" => gen.index,
            "gen_bus" => gen.gen_bus,
            "pg" => gen.pg,
            "qg" => gen.qg,
            "qmax" => gen.qmax,
            "qmin" => gen.qmin,
            "vg" => gen.vg,
            "mbase" => gen.mbase,
            "gen_status" => gen.gen_status,
            "pmax" => gen.pmax,
            "pmin" => gen.pmin,
            "cost" => collect(gen.cost),
            "source_id" => ["gen", gen.index],
        )
    end

    for branch in data.branch
        tap = branch.tap == 0.0 ? 1.0 : branch.tap
        pm_data["branch"][string(branch.index)] = Dict{String,Any}(
            "index" => branch.index,
            "f_bus" => branch.f_bus,
            "t_bus" => branch.t_bus,
            "br_r" => branch.br_r,
            "br_x" => branch.br_x,
            "br_b" => branch.br_b,
            "g_fr" => 0.0,
            "b_fr" => branch.br_b / 2.0,
            "g_to" => 0.0,
            "b_to" => branch.br_b / 2.0,
            "rate_a" => branch.rate_a,
            "rate_b" => branch.rate_b,
            "rate_c" => branch.rate_c,
            "tap" => tap,
            "shift" => branch.shift,
            "br_status" => branch.br_status,
            "angmin" => branch.angmin,
            "angmax" => branch.angmax,
            "transformer" => tap != 1.0,
            "source_id" => ["branch", branch.index],
        )
    end

    for load in data.load
        pm_data["load"][string(load.index)] = Dict{String,Any}(
            "index" => load.index,
            "load_bus" => load.load_bus,
            "pd" => load.pd,
            "qd" => load.qd,
            "status" => load.status,
            "source_id" => ["load", load.index],
        )
    end

    for shunt in data.shunt
        pm_data["shunt"][string(shunt.index)] = Dict{String,Any}(
            "index" => shunt.index,
            "shunt_bus" => shunt.shunt_bus,
            "gs" => shunt.gs,
            "bs" => shunt.bs,
            "status" => shunt.status,
            "source_id" => ["shunt", shunt.index],
        )
    end

    return pm_data
end


@inline function _cost_tuple(cost)
    length(cost) >= 3 && return (Float64(cost[end-2]), Float64(cost[end-1]), Float64(cost[end]))
    length(cost) == 2 && return (0.0, Float64(cost[1]), Float64(cost[2]))
    length(cost) == 1 && return (0.0, 0.0, Float64(cost[1]))
    return (0.0, 0.0, 0.0)
end


function _parsedcase_from_pm_data(pm_data::Dict{String,<:Any})::ParsedCase
    bus_ids = sort(parse.(Int, collect(keys(pm_data["bus"]))))
    gen_ids = sort(parse.(Int, collect(keys(pm_data["gen"]))))
    branch_ids = sort(parse.(Int, collect(keys(pm_data["branch"]))))
    load_ids = sort(parse.(Int, collect(keys(get(pm_data, "load", Dict{String,Any}())))))
    shunt_ids = sort(parse.(Int, collect(keys(get(pm_data, "shunt", Dict{String,Any}())))))

    buses = ParsedBus[
        let bus = pm_data["bus"][string(id)]
            ParsedBus(
                bus["bus_i"], bus["bus_type"],
                get(bus, "pd", 0.0), get(bus, "qd", 0.0),
                get(bus, "gs", 0.0), get(bus, "bs", 0.0),
                get(bus, "area", 1), get(bus, "vm", 1.0), get(bus, "va", 0.0),
                get(bus, "base_kv", 1.0), get(bus, "zone", 1),
                get(bus, "vmax", 1.1), get(bus, "vmin", 0.9)
            )
        end
        for id in bus_ids
    ]

    gens = ParsedGen[
        let gen = pm_data["gen"][string(id)]
            ParsedGen(
                gen["index"], gen["gen_bus"],
                get(gen, "pg", 0.0), get(gen, "qg", 0.0),
                get(gen, "qmax", 0.0), get(gen, "qmin", 0.0),
                get(gen, "vg", 1.0), get(gen, "mbase", pm_data["baseMVA"]),
                get(gen, "gen_status", 1), get(gen, "pmax", 0.0), get(gen, "pmin", 0.0),
                _cost_tuple(get(gen, "cost", Float64[]))
            )
        end
        for id in gen_ids
    ]

    branches = ParsedBranch[
        let branch = pm_data["branch"][string(id)]
            ParsedBranch(
                branch["index"], branch["f_bus"], branch["t_bus"],
                get(branch, "br_r", 0.0), get(branch, "br_x", 0.0),
                get(branch, "br_b", get(branch, "b_fr", 0.0) + get(branch, "b_to", 0.0)),
                get(branch, "rate_a", 0.0), get(branch, "rate_b", 0.0), get(branch, "rate_c", 0.0),
                get(branch, "tap", 1.0), get(branch, "shift", 0.0),
                get(branch, "br_status", 1), get(branch, "angmin", -π), get(branch, "angmax", π)
            )
        end
        for id in branch_ids
    ]

    loads = ParsedLoad[
        let load = pm_data["load"][string(id)]
            ParsedLoad(
                load["index"], load["load_bus"],
                get(load, "pd", 0.0), get(load, "qd", 0.0), get(load, "status", 1)
            )
        end
        for id in load_ids
    ]

    shunts = ParsedShunt[
        let shunt = pm_data["shunt"][string(id)]
            ParsedShunt(
                shunt["index"], shunt["shunt_bus"],
                get(shunt, "gs", 0.0), get(shunt, "bs", 0.0), get(shunt, "status", 1)
            )
        end
        for id in shunt_ids
    ]

    return ParsedCase(
        get(pm_data, "name", "no_name_found"),
        get(pm_data, "source_version", "0.0.0+"),
        get(pm_data, "baseMVA", 1.0),
        buses, gens, branches, loads, shunts
    )
end


function _parse_matpower_typed(data_string::String)::ParsedCase
    lines = split(replace(data_string, "\r\n" => "\n"), '\n')
    current_key = ""
    in_array = false
    row_num = 1

    name = "no_name_found"
    source_version = "0.0.0+"
    baseMVA = 1.0

    buses = ParsedBus[]
    gens = ParsedGen[]
    branches = ParsedBranch[]
    gencost_rows = Vector{Vector{Float64}}()

    for raw_line in lines
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "%") && continue

        if in_array
            startswith(line, "];") && begin
                current_key = ""
                in_array = false
                row_num = 1
                continue
            end

            if current_key == "bus"
                push!(buses, _parse_bus_row(line, baseMVA))
            elseif current_key == "gen"
                push!(gens, _parse_gen_row(line, row_num, baseMVA))
            elseif current_key == "branch"
                push!(branches, _parse_branch_row(line, row_num, baseMVA))
            elseif current_key == "gencost"
                push!(gencost_rows, _parse_cost_row(line))
            end

            row_num += 1
            continue
        end

        startswith(line, "function") && begin
            name = something(_parse_function_name(line), "no_name_found")
            continue
        end

        line_no_comment = strip(_strip_comment(line))
        isempty(line_no_comment) && continue
        assignment = match(r"^mpc\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line_no_comment)
        isnothing(assignment) && continue

        key = assignment.captures[1]
        rhs = strip(assignment.captures[2])

        if key == "version"
            source_version = String(_parse_scalar_rhs(rhs))
        elseif key == "baseMVA"
            baseMVA = Float64(_parse_scalar_rhs(rhs))
        elseif startswith(rhs, "[")
            current_key = key
            in_array = true
            row_num = 1
        end
    end

    isempty(buses) && return ParsedCase(name, source_version, baseMVA, buses, gens, branches, ParsedLoad[], ParsedShunt[])

    buses = _normalize_buses(buses, gens)
    gens = _apply_generator_costs(gens, gencost_rows, baseMVA)
    loads, shunts = _build_bus_injections(buses)
    buses = _clear_bus_injections(buses)

    return ParsedCase(name, source_version, baseMVA, buses, gens, branches, loads, shunts)
end


@inline @views function _parse_bus_row(line::AbstractString, baseMVA::Float64)
    bus_i, bus_type, pd, qd, gs, bs, area, vm, va, base_kv, zone, vmax, vmin =
        @iter_to_ntuple 13 WordedString(SubString(line), ncodeunits(line)) (
            Int, Int, Float64, Float64, Float64, Float64, Int,
            Float64, Float64, Float64, Int, Float64, Float64
        )
    return ParsedBus(
        bus_i, bus_type,
        pd / baseMVA, qd / baseMVA,
        gs / baseMVA, bs / baseMVA,
        area, vm, deg2rad(va), base_kv, zone, vmax, vmin
    )
end


@inline @views function _parse_gen_row(line::AbstractString, index::Int, baseMVA::Float64)
    gen_bus, pg, qg, qmax, qmin, vg, mbase, gen_status, pmax, pmin =
        @iter_to_ntuple 10 WordedString(SubString(line), ncodeunits(line)) (
            Int, Float64, Float64, Float64, Float64, Float64, Float64, Int, Float64, Float64
        )
    return ParsedGen(
        index, gen_bus,
        pg / baseMVA, qg / baseMVA, qmax / baseMVA, qmin / baseMVA,
        vg, mbase, gen_status, pmax / baseMVA, pmin / baseMVA,
        (0.0, 0.0, 0.0)
    )
end


@inline @views function _parse_branch_row(line::AbstractString, index::Int, baseMVA::Float64)
    f_bus, t_bus, br_r, br_x, br_b, rate_a, rate_b, rate_c, tap, shift, br_status, angmin, angmax =
        @iter_to_ntuple 13 WordedString(SubString(line), ncodeunits(line)) (
            Int, Int, Float64, Float64, Float64, Float64, Float64, Float64,
            Float64, Float64, Int, Float64, Float64
        )
    angmin_rad, angmax_rad = _normalize_angle_bounds(deg2rad(angmin), deg2rad(angmax))
    return ParsedBranch(
        index, f_bus, t_bus, br_r, br_x, br_b, rate_a / baseMVA, rate_b / baseMVA, rate_c / baseMVA,
        tap, deg2rad(shift), br_status, angmin_rad, angmax_rad
    )
end


@inline function _parse_cost_row(line::AbstractString)
    row = _parse_numeric_row_ws(line)
    return [Float64(x) for x in row]
end


function _normalize_buses(buses::Vector{ParsedBus}, gens::Vector{ParsedGen})
    normalized = copy(buses)
    has_active_gen = Dict(bus.bus_i => false for bus in buses)
    slack_found = false
    biggest_gen_bus = nothing
    biggest_gen_pmax = -Inf

    for gen in gens
        if gen.gen_status != 0
            has_active_gen[gen.gen_bus] = true
            if gen.pmax > biggest_gen_pmax
                biggest_gen_pmax = gen.pmax
                biggest_gen_bus = gen.gen_bus
            end
        end
    end

    for i in eachindex(normalized)
        bus = normalized[i]
        new_type = bus.bus_type
        if get(has_active_gen, bus.bus_i, false) && bus.bus_type == 1
            new_type = 2
        elseif !get(has_active_gen, bus.bus_i, false) && (bus.bus_type == 2 || bus.bus_type == 3)
            bus.bus_type == 3 && (slack_found = false)
            new_type = 1
        elseif bus.bus_type == 3 && get(has_active_gen, bus.bus_i, false)
            slack_found = true
        end
        normalized[i] = ParsedBus(
            bus.bus_i, new_type, bus.pd, bus.qd, bus.gs, bus.bs, bus.area, bus.vm,
            bus.va, bus.base_kv, bus.zone, bus.vmax, bus.vmin
        )
    end

    if !slack_found && biggest_gen_bus !== nothing
        idx = findfirst(bus -> bus.bus_i == biggest_gen_bus, normalized)
        if !isnothing(idx)
            bus = normalized[idx]
            normalized[idx] = ParsedBus(
                bus.bus_i, 3, bus.pd, bus.qd, bus.gs, bus.bs, bus.area, bus.vm,
                bus.va, bus.base_kv, bus.zone, bus.vmax, bus.vmin
            )
        end
    end

    return normalized
end


function _normalize_angle_bounds(angmin::Float64, angmax::Float64)
    default_pad = deg2rad(60.0)
    if angmin <= -pi / 2
        angmin = -default_pad
    end
    if angmax >= pi / 2
        angmax = default_pad
    end
    if angmin == 0.0 && angmax == 0.0
        angmin = -default_pad
        angmax = default_pad
    end
    return angmin, angmax
end


function _apply_generator_costs(gens::Vector{ParsedGen}, gencost_rows::Vector{Vector{Float64}}, baseMVA::Float64)
    isempty(gencost_rows) && return gens

    out = copy(gens)
    ngen = length(gens)
    ncost = min(ngen, length(gencost_rows))
    for i in 1:ncost
        c = _parse_cost_tuple(gencost_rows[i], baseMVA)
        gen = gens[i]
        out[i] = ParsedGen(
            gen.index, gen.gen_bus, gen.pg, gen.qg, gen.qmax, gen.qmin, gen.vg,
            gen.mbase, gen.gen_status, gen.pmax, gen.pmin, c
        )
    end
    return out
end


function _parse_cost_tuple(cost_row::Vector{Float64}, baseMVA::Float64)
    length(cost_row) < 4 && return (0.0, 0.0, 0.0)
    model = Int(round(cost_row[1]))
    n = Int(round(cost_row[4]))
    first_coeff = 5
    last_coeff = min(length(cost_row), first_coeff + n - 1)
    coeffs = cost_row[first_coeff:last_coeff]

    if model == 2
        coeffs = [baseMVA ^ (n - j) * coeffs[j] for j in eachindex(coeffs)]
    end

    return length(coeffs) >= 3 ? (coeffs[end-2], coeffs[end-1], coeffs[end]) :
           length(coeffs) == 2 ? (0.0, coeffs[1], coeffs[2]) :
           length(coeffs) == 1 ? (0.0, 0.0, coeffs[1]) :
           (0.0, 0.0, 0.0)
end


function _build_bus_injections(buses::Vector{ParsedBus})
    loads = ParsedLoad[]
    shunts = ParsedShunt[]
    load_idx = 1
    shunt_idx = 1

    for bus in buses
        status = bus.bus_type == 4 ? 0 : 1
        if !(iszero(bus.pd) && iszero(bus.qd))
            push!(loads, ParsedLoad(load_idx, bus.bus_i, bus.pd, bus.qd, status))
            load_idx += 1
        end
        if !(iszero(bus.gs) && iszero(bus.bs))
            push!(shunts, ParsedShunt(shunt_idx, bus.bus_i, bus.gs, bus.bs, status))
            shunt_idx += 1
        end
    end

    return loads, shunts
end


function _clear_bus_injections(buses::Vector{ParsedBus})
    out = copy(buses)
    for i in eachindex(out)
        bus = out[i]
        out[i] = ParsedBus(
            bus.bus_i, bus.bus_type,
            0.0, 0.0, 0.0, 0.0,
            bus.area, bus.vm, bus.va, bus.base_kv, bus.zone, bus.vmax, bus.vmin
        )
    end
    return out
end


function _parse_function_name(line::AbstractString)
    m = match(r"^function\s+[^=]+=\s*([A-Za-z_][A-Za-z0-9_]*)", line)
    return isnothing(m) ? nothing : m.captures[1]
end


function _strip_comment(line::AbstractString)
    io = IOBuffer()
    in_string = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '\''
            if in_string && i < lastindex(line) && line[nextind(line, i)] == '\''
                write(io, '\'')
                i = nextind(line, nextind(line, i))
                continue
            end
            in_string = !in_string
            write(io, c)
        elseif c == '%' && !in_string
            break
        else
            write(io, c)
        end
        i = nextind(line, i)
    end
    return String(take!(io))
end


function _parse_scalar_rhs(rhs::AbstractString)
    value = strip(replace(rhs, ";" => ""))
    if startswith(value, "'") && endswith(value, "'")
        return replace(value[2:end-1], "''" => "'")
    end
    return _parse_atom(value)
end


@inline @views function _parse_numeric_row_ws(line::AbstractString)
    row = Any[]
    ws = WordedString(SubString(line), lastindex(line))
    state = iter_ws(ws, 1)
    while state[2] != 0
        isempty(state[1]) && break
        push!(row, _parse_atom(state[1]))
        state = iter_ws(ws, state[2])
    end
    return row
end


function _parse_atom(token::AbstractString; prefer_float::Bool=false)
    value = strip(token)
    isempty(value) && return ""

    lower = lowercase(value)
    lower == "nan" && return NaN
    lower == "inf" && return Inf
    lower == "+inf" && return Inf
    lower == "-inf" && return -Inf

    if !prefer_float && occursin(r"^[+-]?\d+$", value)
        return parse(Int, value)
    end

    return parse(Float64, value)
end
