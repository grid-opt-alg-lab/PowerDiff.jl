using LazyArtifacts

"""
    get_path(library::Symbol)

Resolve an artifact-backed case library bundled with PowerDiff. Currently only
`:pglib` (PGLib-OPF) is available.
"""
function get_path(library::Symbol)
    library == :pglib && return joinpath(artifact"PGLib_opf", "pglib-opf-23.07")
    throw(ArgumentError("unsupported library $library"))
end
