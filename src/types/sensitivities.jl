# =============================================================================
# Sensitivity Result Types
# =============================================================================
#
# Main sensitivity type with symbol metadata fields:
#   Sensitivity{T} carries formulation/operand/parameter as Symbol fields
#
# DC OPF sensitivities use cached KKT derivatives for efficiency.
# AC power flow types are kept for voltage/current sensitivity functions.

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
- `parameter::Symbol`: Parameter tag (:d, :z, :cq, :cl, :fmax, :b, :p, :q)
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
                          row_to_id, id_to_row, col_to_id, id_to_col)
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

# =============================================================================
# DC Power Flow Bundled Types (kept for DCPowerFlowState which doesn't use OPF cache)
# =============================================================================

"""
    DCPFDemandSens

Bundled DC power flow demand sensitivity matrices.
Used by DCPowerFlowState (not DCOPFProblem).
"""
struct DCPFDemandSens <: AbstractSensitivity
    dva_dd::Matrix{Float64}
    df_dd::Matrix{Float64}
end

"""
    DCPFSwitchingSens

Bundled DC power flow switching sensitivity matrices.
Used by DCPowerFlowState (not DCOPFProblem).
"""
struct DCPFSwitchingSens <: AbstractSensitivity
    dva_dz::Matrix{Float64}
    df_dz::Matrix{Float64}
end

"""
    ACOPFSwitchingSens

Bundled AC OPF switching sensitivity matrices.
Used by calc_sensitivity_switching(::ACOPFProblem).
"""
struct ACOPFSwitchingSens <: AbstractSensitivity
    dvm_dz::Matrix{Float64}
    dva_dz::Matrix{Float64}
    dpg_dz::Matrix{Float64}
    dqg_dz::Matrix{Float64}
end

# =============================================================================
# AC Power Flow Sensitivity Types (kept for voltage/current sensitivity functions)
# =============================================================================

"""
    VoltagePowerSensitivity <: AbstractSensitivityPower

Sensitivity of bus voltages with respect to power injections.

# Fields
- `∂v_∂p`: Complex phasor sensitivity dv/dp (n x n)
- `∂v_∂q`: Complex phasor sensitivity dv/dq (n x n)
- `∂vm_∂p`: Magnitude sensitivity d|v|/dp (n x n)
- `∂vm_∂q`: Magnitude sensitivity d|v|/dq (n x n)
"""
struct VoltagePowerSensitivity <: AbstractSensitivityPower
    ∂v_∂p::Matrix{ComplexF64}
    ∂v_∂q::Matrix{ComplexF64}
    ∂vm_∂p::Matrix{Float64}
    ∂vm_∂q::Matrix{Float64}
end

# Non-unicode accessors for VoltagePowerSensitivity
function Base.getproperty(s::VoltagePowerSensitivity, name::Symbol)
    if name === :dvm_dp
        return getfield(s, :∂vm_∂p)
    elseif name === :dvm_dq
        return getfield(s, :∂vm_∂q)
    elseif name === :dv_dp
        return getfield(s, :∂v_∂p)
    elseif name === :dv_dq
        return getfield(s, :∂v_∂q)
    else
        return getfield(s, name)
    end
end

"""
    VoltageTopologySensitivity <: AbstractSensitivityTopology

Sensitivity of bus voltage magnitudes with respect to topology (admittance) parameters.

# Fields
- `∂vm_∂g`: Sensitivity d|v|/dG (n x num_edges)
- `∂vm_∂b`: Sensitivity d|v|/dB (n x num_edges)
"""
struct VoltageTopologySensitivity <: AbstractSensitivityTopology
    ∂vm_∂g::Matrix{Float64}
    ∂vm_∂b::Matrix{Float64}
end

# Non-unicode accessors for VoltageTopologySensitivity
function Base.getproperty(s::VoltageTopologySensitivity, name::Symbol)
    if name === :dvm_dg
        return getfield(s, :∂vm_∂g)
    elseif name === :dvm_db
        return getfield(s, :∂vm_∂b)
    else
        return getfield(s, name)
    end
end

"""
    CurrentPowerSensitivity <: AbstractSensitivityPower

Sensitivity of branch currents with respect to power injections.

# Fields
- `∂I_∂p`: Complex current phasor sensitivity dI/dp (m x n)
- `∂I_∂q`: Complex current phasor sensitivity dI/dq (m x n)
- `∂Im_∂p`: Current magnitude sensitivity d|I|/dp (m x n)
- `∂Im_∂q`: Current magnitude sensitivity d|I|/dq (m x n)
"""
struct CurrentPowerSensitivity <: AbstractSensitivityPower
    ∂I_∂p::Matrix{ComplexF64}
    ∂I_∂q::Matrix{ComplexF64}
    ∂Im_∂p::Matrix{Float64}
    ∂Im_∂q::Matrix{Float64}
end

# Non-unicode accessors for CurrentPowerSensitivity
function Base.getproperty(s::CurrentPowerSensitivity, name::Symbol)
    if name === :dIm_dp
        return getfield(s, :∂Im_∂p)
    elseif name === :dIm_dq
        return getfield(s, :∂Im_∂q)
    elseif name === :dI_dp
        return getfield(s, :∂I_∂p)
    elseif name === :dI_dq
        return getfield(s, :∂I_∂q)
    else
        return getfield(s, name)
    end
end

"""
    CurrentTopologySensitivity <: AbstractSensitivityTopology

Sensitivity of branch currents with respect to topology (admittance) parameters.

# Fields
- `∂Im_∂g`: Sensitivity d|I|/dG (m x num_edges)
- `∂Im_∂b`: Sensitivity d|I|/dB (m x num_edges)
"""
struct CurrentTopologySensitivity <: AbstractSensitivityTopology
    ∂Im_∂g::Matrix{Float64}
    ∂Im_∂b::Matrix{Float64}
end

# Non-unicode accessors for CurrentTopologySensitivity
function Base.getproperty(s::CurrentTopologySensitivity, name::Symbol)
    if name === :dIm_dg
        return getfield(s, :∂Im_∂g)
    elseif name === :dIm_db
        return getfield(s, :∂Im_∂b)
    else
        return getfield(s, name)
    end
end

