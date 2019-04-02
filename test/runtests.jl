using StateEstimation
using LinearAlgebra
using StaticArrays
using LinearAlgebra, StaticArrays, SparseArrays
using LightGraphs, MetaGraphs
using Vec
using Test
using Logging

# Set logging level
global_logger(SimpleLogger(stderr, Logging.Debug))

# Check equality of two arrays
@inline function array_isapprox(x::AbstractArray{F},
                  y::AbstractArray{F};
                  rtol::F=sqrt(eps(F)),
                  atol::F=zero(F)) where {F<:AbstractFloat}
    # Easy check on matching size
    if length(x) != length(y)
        return false
    end
    for (a,b) in zip(x,y)
        if !(isapprox(a,b, rtol=rtol, atol=atol))
            return false
        end
    end
    return true
end

# Check if array equals a single value
@inline function array_isapprox(x::AbstractArray{F},
                  y::F;
                  rtol::F=sqrt(eps(F)),
                  atol::F=zero(F)) where {F<:AbstractFloat}
    for a in x
        if !(isapprox(a, y, rtol=rtol, atol=atol))
            return false
        end
    end
    return true
end

# Define package tests
@time @testset "StateEstimation Package Tests" begin
    testdir = joinpath(dirname(@__DIR__), "test")
    @time @testset "StateEstimation.TransitionModelTests" begin
        include(joinpath(testdir, "unit_tests/test_transition_models.jl"))
    end
    @time @testset "StateEstimation.ObservationModelTests" begin
        include(joinpath(testdir, "unit_tests/test_observation_models.jl"))
    end
    @time @testset "StateEstimation.FilterModelTests" begin
        include(joinpath(testdir, "unit_tests/test_filter_models.jl"))
    end
end
