module StateEstimation

using LinearAlgebra
using Distributions
using Parameters

export
    Filter,
    ObservationModel,
    TransitionModel,
    DeterministicObservationModel,
    ProbabilisticObservationModel,
    getDiscreteLinearSystem,
    DiscreteLinearSystem,
    DiscreteLinearGaussianSystem,
    propagate,
    propagate!,
    state_jacobian,

    LinearSensor,
    LinearGaussianSensor,
    RangeSensor,
    GaussianRangeSensor,
    BearingSensor,
    GaussianBearingSensor,
    observe,
    measurement_jacobian

include("observation_models.jl")
include("transition_models.jl")

abstract type Filter end

struct BinaryDiscreteFilter{V,T,O} <: Filter
    pts                 ::Vector{V} # distribution over possible locations of target
    x                   ::Vector{Bool} # hypothesis
    transition_model    ::T
    observation_model   ::O
end

end # module
