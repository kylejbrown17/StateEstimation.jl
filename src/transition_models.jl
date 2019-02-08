export
    TransitionModel,
    DynamicSystem,
    DiscreteLinearSystem,
    DiscreteLinearGaussianSystem,
    BinaryReachabilityTransitionModel,
    propagate,
    # propagate!,
    state_jacobian

abstract type TransitionModel end
abstract type DeterministicTransitionModel <: TransitionModel end
abstract type ProbabilisticTransitionModel <: TransitionModel end

deterministic(m::DeterministicTransitionModel) = m
# deterministic(m::ProbabilisticTransitionModel) = throw(deterministic, string("deterministic(m::",typeof(m),"not defined)"))

mutable struct DynamicSystem{T,F}
    x::T                # system state
    transition_model::F # state transition_model model
end
deterministic(m::DynamicSystem{T,F} where {T,F}) = DynamicSystem(x,deterministic(m.transition_model))
predict(sys,u) = predict(sys.transition_model,sys.x,u)
function predict!(sys,u)
    sys.x = predict(sys,u)
end

mutable struct DiscreteLinearSystem <: DeterministicTransitionModel
    x::Vector{Float64} # state
    A::Matrix{Float64} # drift matrix
    B::Matrix{Float64} # control matrix
end
# function predict(sys::DiscreteLinearSystem,x,u)
#     sys.A*x + sys.B*u
# end
function propagate(sys::DiscreteLinearSystem,x,u)
    sys.A*x + sys.B*u
end
# function propagate!(sys::DiscreteLinearSystem,u)
#     sys.x = propagate(sys,sys.x,u)
# end
function state_jacobian(sys::DiscreteLinearSystem,x)
    sys.A
end
function state_jacobian(sys::DiscreteLinearSystem)
    sys.A
end

mutable struct DiscreteLinearGaussianSystem <: ProbabilisticTransitionModel
    x::Vector{Float64} # state
    A::Matrix{Float64} # drift matrix
    B::Matrix{Float64} # control matrix
    Q::Matrix{Float64} # process noise
    proc_noise::Distributions.MultivariateNormal
end
deterministic(m::DiscreteLinearGaussianSystem) = DiscreteLinearSystem(m.x,m.A,m.B)
function DiscreteLinearGaussianSystem(x::Vector,A::Matrix,B::Matrix,Q)
    DiscreteLinearGaussianSystem(x,A,B,Q,MultivariateNormal(zeros(size(x)),Q))
end
# function predict(sys::DiscreteLinearGaussianSystem,x,u)
#     sys.A*x + sys.B*u
# end
function propagate(sys::DiscreteLinearGaussianSystem,x,u)
    sys.A*x + sys.B*u + rand(sys.proc_noise)
end
# function propagate!(sys::DiscreteLinearGaussianSystem,u)
#     sys.x = propagate(sys,sys.x,u)
# end
function state_jacobian(sys::DiscreteLinearGaussianSystem,x)
    state_jacobian(sys)
end
function state_jacobian(sys::DiscreteLinearGaussianSystem)
    sys.A
end

mutable struct BinaryReachabilityTransitionModel <: DeterministicTransitionModel
    A::Matrix{Bool}
end
propagate(sys::BinaryReachabilityTransitionModel,x,u) = A*x
