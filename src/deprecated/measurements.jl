# =============================================================================
# DEPRECATED: Measurement Types for State Estimation
# =============================================================================
#
# These types were used for state estimation experiments and are no longer
# actively maintained. They are kept for backwards compatibility.

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
