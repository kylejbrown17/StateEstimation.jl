export
    DynamicSystem,
    DiscreteLinearSystem,
    single_integrator_1D,
    single_integrator_2D,
    blended_single_integrator_2D,
    double_integrator_1D,
    double_integrator_2D,
    DiscreteLinearGaussianSystem,
    BinaryReachabilityTransitionModel,
    propagate,
    # propagate!,
    state_jacobian

# abstract type TransitionModel end
abstract type DeterministicTransitionModel <: StateTransitionModel end
abstract type ProbabilisticTransitionModel <: StateTransitionModel end
deterministic(m::DeterministicTransitionModel) = m
# deterministic(m::ProbabilisticTransitionModel) = throw(deterministic, string("deterministic(m::",typeof(m),"not defined)"))

mutable struct DynamicSystem{T,F}
    x::T                # system state
    transition_model::F # state transition_model model
end
deterministic(m::DynamicSystem) = DynamicSystem(x,deterministic(m.transition_model))

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
function propagate(sys::DiscreteLinearSystem,x,u)
    sys.A*x + sys.B*u
end
function state_jacobian(sys::DiscreteLinearSystem,x)
    sys.A
end
function state_jacobian(sys::DiscreteLinearSystem)
    sys.A
end

"""
    single_integrator_1D

    state = [x]
"""
function single_integrator_1D(dt::Float64)
    A = [1.0]
    B = dt * [1.0]
    DiscreteLinearSystem(A,B)
end

"""
    single_integrator_2D

    state = [x, y]
"""
function single_integrator_2D(dt::Float64)
    A = [
        1.0 0.0;
        0.0 1.0
    ]
    B = dt * [
        1.0 0.0;
        0.0 1.0
    ]
    DiscreteLinearSystem(A,B)
end

"""
    blended_single_integrator_2D

    state = [x, y, ẋ, ẏ]
    `μ` is a blending coefficient that combines the previous velocity with the
    commanded velocity
"""
function blended_single_integrator_2D(dt::Float64,μ::Float64=1.0)
    A = [
        1.0     0.0     (1.0-μ)*dt  0.0         ;
        0.0     1.0     0.0         (1.0-μ)*dt  ;
        0.0     0.0     1.0-μ       0.0         ;
        0.0     0.0     0.0         0.0-μ       ;
    ]
    B = [
        μ*dt    0.0 ;
        0.0     μ*dt;
        μ   0.0 ;
        0.0     μ ;
    ]
    DiscreteLinearSystem(A,B)
end

"""
    double_integrator_2D

    state = [x, ẋ]
"""
function double_integrator_1D(dt::Float64)
    A = [
        1.0 0.0;
        0.0 1.0;
    ]
    B = [
        .5*dt^2 0.0;
        0.0     dt;
    ]
    DiscreteLinearSystem(A,B)
end

"""
    double_integrator_2D

    state = [x, y, ẋ, ẏ]
"""
function double_integrator_2D(dt::Float64)
    A = [
        1.0 0.0 dt 0.0;
        0.0 1.0 0.0 dt;
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0
    ]
    B = [
        .5*dt^2 0.0;
        0.0     .5*dt^2;
        dt      0.0;
        0.0     dt;
    ]
    DiscreteLinearSystem(A,B)
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
function DiscreteLinearGaussianSystem(sys::DiscreteLinearSystem,Q::SMatrix)
    return DiscreteLinearGaussianSystem(sys.A,sys.B,Q)
end

function propagate(sys::DiscreteLinearGaussianSystem,x,u)
    sys.A*x + sys.B*u + rand(sys.proc_noise)
end
function state_jacobian(sys::DiscreteLinearGaussianSystem,x)
    state_jacobian(sys)
end
function state_jacobian(sys::DiscreteLinearGaussianSystem)
    sys.A
end

# mutable struct BinaryReachabilityTransitionModel{N} <: DeterministicTransitionModel
#     A::SMatrix{N,N,Bool}
# end
mutable struct BinaryReachabilityTransitionModel{T <: AbstractMatrix{Int}} <: DeterministicTransitionModel
    A::T
end
propagate(sys::BinaryReachabilityTransitionModel,x,u) = typeof(x)(sys.A*x .> 0)
