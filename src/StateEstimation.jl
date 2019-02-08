module StateEstimation

using LinearAlgebra
using Distributions
using Parameters
using StaticArrays

export
    Filter,
    BinaryDiscreteFilter,
    KalmanFilter,
    ExtendedKalmanFilter, EKF,
    UnscentedKalmanFilter, UKF,
    deterministic,
    predict,
    predict!,
    update,
    update!


include("observation_models.jl")
include("transition_models.jl")

abstract type Filter end
function deterministic(m::T where T)
    error(string("deterministic(m::",typeof(m),") not implemented for "))
end

struct BinaryDiscreteFilter{V,T,O} <: Filter
    pts                 ::Vector{V} # distribution over possible locations of target
    x                   ::Vector{Bool} # hypothesis
    transition_model    ::T
    observation_model   ::O
end

mutable struct KalmanFilter{T,V} <: Filter
    μ::T # μ::SVector{n,T} # mean vector
    Σ::V # Σ::SMatrix{n,n,T} covariance matrix
    # A::SMatrix{n,n,T}
    # B::SMatrix{n,m,T}
    # C::SMatrix{p,n,T}
    # D::SMatrix{p,m,T}
    # Q::SMatrix{n,n,Float64}
    # R::SMatrix{p,p,Float64}
    transition_model::DiscreteLinearGaussianSystem
    observation_model::LinearGaussianSensor
end
function KalmanFilter(μ,Σ,A,B,C,D,Q,R)
    KalmanFilter(μ,Σ,DiscreteLinearGaussianSystem(A,B,Q),LinearGaussianSensor(C,D,R))
end
function KalmanFilter(μ,Σ,A,B,C,D,Q,R)
    KalmanFilter(μ,Σ,DiscreteLinearGaussianSystem(A,B,Q),LinearGaussianSensor(C,D,R))
end
function predict(m::KalmanFilter,μ,Σ,u::T where T)
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
function predict!(m::KalmanFilter,u::T where T)
    """
        Prediction step in Kalman Filter
    """
    m.μ,m.Σ = predict(m,m.u,m.Σ)
end
function update(m::KalmanFilter,μ,Σ,z::T where T)
    C = m.observation_model.C
    D = m.observation_model.D
    R = m.observation_model.R

    Kt = Σ*C'*inv(C*Σ*C' + R)

    μ = μ + Kt*(z - C*μ)
    Σ = Σ - Kt*C*Σ

    return μ,Σ
end
function update!(m::KalmanFilter,z::T where T)
    m.μ,m.Σ = update(m,m.μ,m.Σ,z)
end

mutable struct ExtendedKalmanFilter{n,m,p,F,G,T} <: Filter
    μ::SVector{n,T} # mean vector
    Σ::SMatrix{n,n,T} # covariance matrix
    Q::SMatrix{n,n,Float64} # process noise
    R::SMatrix{p,p,Float64} # measurement noise
    transition_model::F
    observation_model::G
end
const EKF = ExtendedKalmanFilter
function predict(m::EKF,μ,Σ,u::T where T)
    At = state_jacobian(m.transition_model,μ)
    Q = m.Q

    μ = propagate(deterministic(m.transition_model),μ,u)
    Σ = At*Σ*At' + Q

    return μ,Σ
end
function predict!(m::EKF,u::T where T)
    m.μ,m.Σ = predict(m,m.μ,m.Σ,u)
end
function update(m::EKF,μ,Σ,z::T where T)
    Ct = measurement_jacobian(m.observation_model, μ)
    R = m.R

    Kt = Σ*Ct'*inv(C*Σ*Ct' + R)

    μ = μ - Kt*(z - observe(deterministi(m.transition_model),μ))
    Σ = Σ - Kt*Ct*Σ
end
function update!(m::EKF,z::T where T)
    m.μ,m.Σ = update!(m,m.μ,m.Σ,z)
end

mutable struct UnscentedKalmanFilter{n,m,p,F,G,T} <: Filter
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
function unscented_transform(μ,Σ,λ,n)
    σ_pts = hcat([μ, μ+sqrtm(Σ), μ-sqrtm(Σ)]...)
    weights = ones(2*n+1) * 1.0 / (n + λ)
    weights[1] *= 2
    return σ_pts, weights
end
function inverse_unscented_transform(σ_pts,weights)
    μ = σ_pts * weights
    Δ = σ_pts .- μ
    Σ = Δ * diagm(weights) * Δ'
end
function predict(m::UKF,μ,Σ,u::T where T)
    σ_pts, weights = unscented_transform(μ,Σ,m.λ,m.n)
    σ_pts = hcat([propagate(deterministic(m.transition_model),σ_pts[:,i],u) for i in 1:size(σ_pts,2)]...)
    (μ,Σ) = inverse_unscented_transform(σ_pts,weights)
    Σ = Σ + m.Q
    return μ,Σ
end
function update(m::UKF,μ,Σ,z::T where T)
    σ_pts, weights = unscented_transform(μ,Σ,m.λ,m.n)
    Z = hcat([observe(deterministic(m.observation_model),σ_pts[:,i]) for i in 1:size(σ_pts,2)]...)
    Ẑ = Z * weights
    W = diagm(weights)
    Szz = (Ẑ .- z)*W*(Ẑ .- z) + m.R
    Sxz = (σ_pts .- μ) * W * (Ẑ .- z)
    Kt = Sxz*inv(Szz)

    μ = Kt*(z .- Ẑ)
    Σ = Σ - Kt*Szz*Kt'

    return μ,Σ
end

# mutable struct MultiHypothesisKalmanFilter <: Filter
# end





end # module
