export
    TransitionModel,
    getDiscreteLinearSystem,
    DiscreteLinearSystem,
    DiscreteLinearGaussianSystem,
    BinaryReachabilityTransitionModel,
    propagate,
    propagate!,
    state_jacobian

abstract type TransitionModel end

mutable struct DiscreteLinearSystem <: TransitionModel
    x::Vector{Float64} # state
    A::Matrix{Float64} # drift matrix
    B::Matrix{Float64} # control matrix
end
function propagate(sys::DiscreteLinearSystem,x,u)
    sys.A*x + sys.B*u
end
function propagate!(sys::DiscreteLinearSystem,u)
    sys.x = propagate(sys,sys.x,u)
end
function state_jacobian(sys::DiscreteLinearSystem)
    sys.A
end

mutable struct DiscreteLinearGaussianSystem <: TransitionModel
    x::Vector{Float64} # state
    A::Matrix{Float64} # drift matrix
    B::Matrix{Float64} # control matrix
    Q::Matrix{Float64} # process noise
    proc_noise::Distributions.MultivariateNormal
end
function DiscreteLinearGaussianSystem(x::Vector,A::Matrix,B::Matrix,Q)
    DiscreteLinearGaussianSystem(x,A,B,Q,MultivariateNormal(zeros(size(x)),Q))
end
function propagate(sys::DiscreteLinearGaussianSystem,x,u)
    sys.A*x + sys.B*u + rand(sys.proc_noise)
end
function propagate!(sys::DiscreteLinearGaussianSystem,u)
    sys.x = propagate(sys,sys.x,u)
end
function state_jacobian(sys::DiscreteLinearGaussianSystem)
    sys.A
end

mutable struct BinaryReachabilityTransitionModel
    A::Matrix{Bool}
end
propagate(sys::BinaryReachabilityTransitionModel,x,u) = A*x
