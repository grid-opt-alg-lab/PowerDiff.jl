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
