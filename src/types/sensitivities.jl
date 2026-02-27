# =============================================================================
# Sensitivity Result Type
# =============================================================================
#
# Main sensitivity type with symbol metadata fields:
#   Sensitivity{T} carries formulation/operand/parameter as Symbol fields
#
# Internal sensitivity functions return named tuples, which the interface
# layer wraps into Sensitivity{T} for the public API.

# =============================================================================
# Main Sensitivity Type
# =============================================================================

"""
    Sensitivity{T} <: AbstractMatrix{T}

Sensitivity matrix with symbol metadata for formulation, operand, and parameter.

Implements the AbstractMatrix interface for seamless matrix operations while
providing bidirectional index mappings between internal indices and element IDs.

# Type Parameter
- `T <: Number`: Element type (Float64, ComplexF64)

# Fields
- `matrix::Matrix{T}`: The sensitivity matrix data
- `formulation::Symbol`: Formulation tag (:dcpf, :dcopf, :acpf, :acopf)
- `operand::Symbol`: Operand tag (:va, :vm, :pg, :qg, :f, :lmp, :im, :v)
- `parameter::Symbol`: Parameter tag (:d, :sw, :cq, :cl, :fmax, :b, :p, :q)
- `row_to_id::Vector{Int}`: Maps internal row index → external element ID
- `id_to_row::Dict{Int,Int}`: Maps external element ID → internal row index
- `col_to_id::Vector{Int}`: Maps internal col index → external element ID
- `id_to_col::Dict{Int,Int}`: Maps external element ID → internal col index

# Examples
```julia
sens = calc_sensitivity(prob, :lmp, :d)
sens.formulation  # :dcopf
sens.operand      # :lmp
sens.parameter    # :d
size(sens)        # (n, n)
sens[2, 3]        # Access element
Matrix(sens)      # Get raw matrix
```
"""
struct Sensitivity{T <: Number} <: AbstractMatrix{T}
    matrix::Matrix{T}
    formulation::Symbol
    operand::Symbol
    parameter::Symbol
    row_to_id::Vector{Int}
    id_to_row::Dict{Int,Int}
    col_to_id::Vector{Int}
    id_to_col::Dict{Int,Int}
end

# Convenience constructor from mappings tuple — infers T from matrix element type
function Sensitivity(
    matrix::Matrix{T},
    formulation::Symbol,
    operand::Symbol,
    parameter::Symbol,
    row_mapping::Tuple{Vector{Int}, Dict{Int,Int}},
    col_mapping::Tuple{Vector{Int}, Dict{Int,Int}}
) where {T <: Number}
    row_to_id, id_to_row = row_mapping
    col_to_id, id_to_col = col_mapping
    return Sensitivity{T}(matrix, formulation, operand, parameter,
                          copy(row_to_id), copy(id_to_row),
                          copy(col_to_id), copy(id_to_col))
end

# =============================================================================
# AbstractMatrix Interface
# =============================================================================

Base.size(s::Sensitivity) = size(s.matrix)
Base.size(s::Sensitivity, d::Integer) = size(s.matrix, d)
Base.getindex(s::Sensitivity, i::Int) = s.matrix[i]
Base.getindex(s::Sensitivity, i::Int, j::Int) = s.matrix[i, j]
Base.getindex(s::Sensitivity, I...) = s.matrix[I...]

# Enable matrix arithmetic
Base.Matrix(s::Sensitivity) = s.matrix
Base.:*(a::Number, s::Sensitivity) = a * s.matrix
Base.:*(s::Sensitivity, a::Number) = s.matrix * a
Base.:*(s::Sensitivity, v::AbstractVector) = s.matrix * v
Base.:*(m::AbstractMatrix, s::Sensitivity) = m * s.matrix

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", s::Sensitivity{T}) where {T}
    if T === Float64
        print(io, "Sensitivity(:$(s.formulation), :$(s.operand), :$(s.parameter))")
    else
        print(io, "Sensitivity{$T}(:$(s.formulation), :$(s.operand), :$(s.parameter))")
    end
    println(io, " $(size(s, 1))×$(size(s, 2)):")
    Base.print_matrix(io, s.matrix)
end

function Base.show(io::IO, s::Sensitivity{T}) where {T}
    if T === Float64
        print(io, "Sensitivity(:$(s.formulation), :$(s.operand), :$(s.parameter), $(size(s, 1))×$(size(s, 2)))")
    else
        print(io, "Sensitivity{$T}(:$(s.formulation), :$(s.operand), :$(s.parameter), $(size(s, 1))×$(size(s, 2)))")
    end
end
