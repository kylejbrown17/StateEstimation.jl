module StateEstimation

using LinearAlgebra
using Distributions
using Parameters
using StaticArrays
using NearestNeighbors

export
    Filter,
    BinaryDiscreteFilter,
    KalmanFilter,
    ExtendedKalmanFilter, EKF,
    UnscentedKalmanFilter, UKF,
    unscented_transform,
    inverse_unscented_transform,
    deterministic,
    predict,
    predict!,
    update,
    update!


const MatrixLike = Union{Matrix,Vector,Real}

include("observation_models.jl")
include("transition_models.jl")

abstract type Filter end
function deterministic(m)
    error(string("deterministic(m::",typeof(m),") not implemented for "))
end

mutable struct KalmanFilter{N,T} <: Filter
    μ::SVector{N,T} # μ::SVector{n,T} # mean vector
    Σ::SMatrix{N,N,T} # Σ::SMatrix{n,n,T} covariance matrix
    # A::SMatrix{n,n,T}
    # B::SMatrix{n,m,T}
    # C::SMatrix{p,n,T}
    # D::SMatrix{p,m,T}
    # Q::SMatrix{n,n,Float64}
    # R::SMatrix{p,p,Float64}
    transition_model::DiscreteLinearGaussianSystem
    observation_model::LinearGaussianSensor
end
const KF = KalmanFilter
function KalmanFilter(μ::MatrixLike,Σ::MatrixLike,sysF::TransitionModel,sysG::ObservationModel)
    n = size(μ,1)
    KalmanFilter(SVector{n}(μ),SMatrix{n,n}(Σ),sysF,sysG)
end
function KalmanFilter(μ,Σ,A,B,C,D,Q,R)
    KalmanFilter(μ,Σ,DiscreteLinearGaussianSystem(A,B,Q),LinearGaussianSensor(C,D,R))
end
function predict(m::KalmanFilter,μ,Σ,u)
    """
        Prediction step in Kalman Filter
    """
    A = m.transition_model.A
    B = m.transition_model.B
    Q = m.transition_model.Q

    μ = A*μ + B*u
    Σ = A*Σ*A' + Q

    return μ,Σ
end
function predict!(m::KalmanFilter,u)
    """
        Prediction step in Kalman Filter
    """
    m.μ,m.Σ = predict(m,m.μ,m.Σ,u)
end
function update(m::KalmanFilter,μ,Σ,z)
    C = m.observation_model.C
    D = m.observation_model.D
    R = m.observation_model.R

    Kt = Σ*C'*inv(C*Σ*C' + R)

    μ = μ + Kt*(z - C*μ)
    Σ = Σ - Kt*C*Σ

    return μ,Σ
end
function update!(m::KalmanFilter,z)
    m.μ,m.Σ = update(m,m.μ,m.Σ,z)
end

mutable struct ExtendedKalmanFilter{N,P,F,G,T} <: Filter
    μ::SVector{N,T} # mean vector
    Σ::SMatrix{N,N,T} # covariance matrix
    Q::SMatrix{N,N,Float64} # process noise
    R::SMatrix{P,P,Float64} # measurement noise
    transition_model::F
    observation_model::G
end
function ExtendedKalmanFilter(μ::MatrixLike,Σ::MatrixLike,Q::MatrixLike,R::MatrixLike,sysF,sysG)
    n = size(μ,1)
    p = size(Q,1)
    EKF(SVector{n}(μ),SMatrix{n,n}(Σ),SMatrix{n,n}(Q),SMatrix{p,p}(R),sysF,sysG)
end
const EKF = ExtendedKalmanFilter
function predict(m::EKF,μ,Σ,u)
    At = state_jacobian(m.transition_model,μ)
    Q = m.Q

    μ = propagate(deterministic(m.transition_model),μ,u)
    Σ = At*Σ*At' + Q

    return μ,Σ
end
function predict!(m::EKF,u)
    m.μ,m.Σ = predict(m,m.μ,m.Σ,u)
end
function update(m::EKF,μ,Σ,z)
    Ct = measurement_jacobian(m.observation_model, μ)
    R = m.R

    Kt = Σ*Ct'*inv(Ct*Σ*Ct' + R)

    μ = μ + Kt*(z - observe(deterministic(m.observation_model),μ))
    Σ = Σ - Kt*Ct*Σ
    return μ,Σ
end
function update!(m::EKF,z)
    m.μ,m.Σ = update(m,m.μ,m.Σ,z)
end

mutable struct UnscentedKalmanFilter{n,p,F,G,T} <: Filter
    μ::SVector{n,T} # mean vector
    Σ::SMatrix{n,n,T} # covariance matrix
    Q::SMatrix{n,n,Float64} # process noise
    R::SMatrix{p,p,Float64} # measurement noise
    λ::Float64
    n::Int
    transition_model::F
    observation_model::G
end
const UKF = UnscentedKalmanFilter
function UnscentedKalmanFilter(
    μ::MatrixLike,Σ::MatrixLike,Q::MatrixLike,R::MatrixLike,λ,n,
    sysF::TransitionModel,sysG::ObservationModel)
    N = size(μ,1)
    P = size(R,1)
    UKF(SVector{N}(μ),SMatrix{N,N}(Σ),SMatrix{N,N}(Q),SMatrix{P,P}(R),float(λ),n,sysF,sysG)
end
function unscented_transform(μ,Σ,λ,n)
    σ_pts = hcat([μ, μ.+sqrt(Σ), μ.-sqrt(Σ)]...)
    weights = ones(2*n+1) * 1.0 / (n + λ)
    weights[1] *= 2
    return σ_pts, weights
end
function inverse_unscented_transform(σ_pts,weights)
    μ = σ_pts * weights
    Δ = σ_pts .- μ
    Σ = Δ * diagm((0=>weights)) * Δ'
    return μ,Σ
end
function predict(m::UKF,μ,Σ,u)
    σ_pts, weights = unscented_transform(μ,Σ,m.λ,m.n)
    σ_pts = hcat([propagate(deterministic(m.transition_model),σ_pts[:,i],u) for i in 1:size(σ_pts,2)]...)
    (μ,Σ) = inverse_unscented_transform(σ_pts,weights)
    Σ = Σ + m.Q
    return μ,Σ
end
function predict!(m::UKF,u)
    m.μ,m.Σ = predict(m,m.μ,m.Σ,u)
end
function update(m::UKF,μ,Σ,z)
    σ_pts, weights = unscented_transform(μ,Σ,m.λ,m.n)
    Z = hcat([observe(deterministic(m.observation_model),σ_pts[:,i]) for i in 1:size(σ_pts,2)]...)
    ẑ = Z * weights
    W = diagm((0=>weights))
    Szz = (Z .- z)*W*(Z .- z)' + m.R
    Sxz = (σ_pts .- μ) * W * (Z .- z)'
    Kt = Sxz*inv(Szz)

    μ = μ + Kt*(z .- ẑ)
    Σ = Σ - Kt*Szz*Kt'

    return μ,Σ
end
function update!(m::UKF,u)
    m.μ,m.Σ = update(m,m.μ,m.Σ,u)
end

mutable struct BinaryDiscreteFilter{N,V,F,G} <: Filter
    pts                 ::SVector{N,V} # distribution over possible locations of target
    μ                   ::SVector{N,Bool} # hypothesis
    kdtree              ::NearestNeighbors.KDTree
    n                   ::Int
    transition_model    ::BinaryReachabilityTransitionModel
    observation_model   ::G
end
function predict(m::BinaryDiscreteFilter,μ)
    propagate(m.transition_model,μ)
end
function predict!(m::BinaryDiscreteFilter,μ)
    m.μ = predict(m,μ)
end
function update(m::BinaryDiscreteFilter,μ,z)
    """
        z is a point estimate of the target location in R²
    """
    idxs, dists = get_neighbors(m.kdtree, z, m.n)
    N = size(μ,1)
    SVector{size(μ,1),Bool}([(i ∈ idxs) for i in 1:N])
end
function update!(m::BinaryDiscreteFilter,z)
    m.μ = update(m,m.μ,z)
end
# mutable struct MultiHypothesisFilter{F} <: Filter
#     weights::Vector{Float64}
#     filters::Vector{F}
# end
# const MHF = MultiHypothesisFilter






end # module
