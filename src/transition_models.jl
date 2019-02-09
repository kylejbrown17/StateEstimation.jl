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
deterministic(m::DynamicSystem) = DynamicSystem(x,deterministic(m.transition_model))
# predict(sys,u) = predict(sys.transition_model,sys.x,u)
# function predict!(sys,u)
#     sys.x = predict(sys,u)
# end

mutable struct DiscreteLinearSystem{n,m,T} <: DeterministicTransitionModel
    # x::SVector{n,T} # state
    A::SMatrix{n,n,T} # drift matrix
    B::SMatrix{n,m,T} # control matrix
end
function DiscreteLinearSystem(A::MatrixLike,B::MatrixLike)
    n = size(A,1)
    m = size(B,2)
    DiscreteLinearSystem(SMatrix{n,n}(A),SMatrix{n,m}(B))
end
# function DiscreteLinearSystem(A::Matrix,B::Vector)
#     n = size(A,1)
#     m = 1
#     DiscreteLinearSystem(SMatrix{n,n}(A),SMatrix{n,m}(B))
# end
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

mutable struct DiscreteLinearGaussianSystem{n,m,T} <: ProbabilisticTransitionModel
    # x::Vector{Float64} # state
    A::SMatrix{n,n,T} # drift matrix
    B::SMatrix{n,m,T} # control matrix
    Q::SMatrix{n,n,Float64} # process noise
    proc_noise::Distributions.MultivariateNormal
end
deterministic(m::DiscreteLinearGaussianSystem) = DiscreteLinearSystem(m.A,m.B)
function DiscreteLinearGaussianSystem(A::SMatrix,B::SMatrix,Q::SMatrix)
    DiscreteLinearGaussianSystem(A,B,Q,MultivariateNormal(zeros(size(A,1)),Matrix(Q)))
end
function DiscreteLinearGaussianSystem(A::MatrixLike,B::MatrixLike,Q::MatrixLike)
    n = size(A,1)
    m = size(B,2)
    DiscreteLinearGaussianSystem(
        SMatrix{n,n}(A),
        SMatrix{n,m}(B),
        SMatrix{n,n}(Q))
end
# DiscreteLinearGaussianSystem(A::Matrix,B::Vector,Q::Matrix) = DiscreteLinearGaussianSystem(A,Matrix{length(B),1}(B),Q)
# function DiscreteLinearGaussianSystem(A::Matrix,B::Vector,Q::Matrix)
#     n = size(A,1)
#     m = 1
#     DiscreteLinearGaussianSystem(
#         SMatrix{n,n}(A),
#         SMatrix{n,m}(B),
#         SMatrix{n,n}(Q),
#         MultivariateNormal(zeros(n),Q))
# end
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

mutable struct BinaryReachabilityTransitionModel{N} <: DeterministicTransitionModel
    A::SMatrix{N,N,Bool}
end
propagate(sys::BinaryReachabilityTransitionModel,x,u) = A*x
