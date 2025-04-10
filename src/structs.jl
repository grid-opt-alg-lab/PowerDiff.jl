abstract type AbstractMeasurement end


struct NodalMeasurement <: AbstractMeasurement
    var::Union{String,Int,Symbol}
    val::Float64
    σ::Float64 
    node::Int 
end

struct NetworkMeasurement <: AbstractMeasurement
    var::Union{String,Int,Symbol}
    val::Vector{Float64}
    σ::Union{Vector{Float64},Float64}
    nodes::Vector{Int}
end

struct PowerNetworkMeasurement <: AbstractMeasurement
    p::Union{Vector{NodalMeasurement},NetworkMeasurement}
    q::Union{Vector{NodalMeasurement},NetworkMeasurement}
    v::Union{Vector{NodalMeasurement},NetworkMeasurement}
    δ::Union{Vector{NodalMeasurement},NetworkMeasurement}
end

mutable struct ComplexNetworkState
    G::AbstractVector{Float64}
    B::AbstractVector{Float64}
    Vre::Vector{Float64}
    Vim::Vector{Float64}
end




#TODO
"""
"""
function VectorizedAdmittanceMatrix(Y::PowerModels.AdmittanceMatrix)
    nothing
end

abstract type AbstractSensitivity end
abstract type AbstractSensitivityPower <: AbstractSensitivity end
abstract type AbstractSensitivityToplogy <: AbstractSensitivity end
abstract type AbstractSensitivityTap <: AbstractSensitivity end

struct CurrentSensitivityPower <: AbstractSensitivityPower
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end

struct VoltageSensitivityPower <: AbstractSensitivityPower
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end

struct CurrentSensitivityTopology <: AbstractSensitivityToplogy
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end
                                            
struct VoltageSensitivityTopology <: AbstractSensitivityToplogy
    p::AbstractMatrix{Float64}
    q::AbstractMatrix{Float64}
end


