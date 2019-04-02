module StateEstimation

using LinearAlgebra
using Distributions
using Parameters
using StaticArrays
using NearestNeighbors
using LightGraphs, MetaGraphs
using DecisionMaking
using CoordinateTransformations, Rotations

export
    Filter,
    StateEstimator,
    KalmanFilter,
    ExtendedKalmanFilter, EKF,
    UnscentedKalmanFilter, UKF,
    BinaryDiscreteFilter,
    TimedFilter,
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
const StateEstimator = Filter
function deterministic(m)
    error(string("deterministic(m::",typeof(m),") not implemented for "))
end
predict!(m::Filter,u,t::Float64) = predict!(m,μ)

mutable struct KalmanFilter{N,T <: Real,V <: StaticArray{Tuple{N},T,1}, W <: SMatrix{N,N,T}} <: Filter
    μ::V # μ::SVector{n,T} # mean vector
    Σ::W # Σ::SMatrix{n,n,T} covariance matrix
    transition_model::DiscreteLinearGaussianSystem
    observation_model::LinearGaussianSensor
end
const KF = KalmanFilter
function KalmanFilter(μ::StaticArray,Σ::MatrixLike,sysF,sysG)
    n = size(μ,1)
    KalmanFilter(μ,SMatrix{n,n}(Σ),sysF,sysG)
end
function KalmanFilter(μ::MatrixLike,Σ::MatrixLike,sysF,sysG)
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
    A = m.transition_model.A # state jacobian
    B = m.transition_model.B # control jacobian
    Q = m.transition_model.Q # process noise covariance

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
    C = m.observation_model.sensor.C # measurement state jacobian
    D = m.observation_model.sensor.D # measurement control jacobian
    R = m.observation_model.R # measurement noise covariance

    Kt = Σ*C'*inv(C*Σ*C' + R)

    μ = μ + Kt*(z - C*μ)
    Σ = Σ - Kt*C*Σ

    return μ,Σ
end
function update!(m::KalmanFilter,z)
    m.μ,m.Σ = update(m,m.μ,m.Σ,z)
end

mutable struct ExtendedKalmanFilter{N,P,T <: Real,V <: StaticArray{Tuple{N},T,1}, W <: SMatrix{N,N,T}, F,G} <: Filter
    μ::V # mean vector
    Σ::W # covariance matrix
    Q::SMatrix{N,N,Float64} # process noise
    R::SMatrix{P,P,Float64} # measurement noise
    transition_model::F
    observation_model::G
end
function ExtendedKalmanFilter(μ::StaticArray,Σ::MatrixLike,Q,R,sysF,sysG)
    n = size(μ,1)
    p = size(Q,1)
    EKF(μ,SMatrix{n,n}(Σ),SMatrix{n,n}(Q),SMatrix{p,p}(R),sysF,sysG)
end
function ExtendedKalmanFilter(μ::MatrixLike,Σ::MatrixLike,Q,R,sysF,sysG)
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

mutable struct UnscentedKalmanFilter{N,P,T <: Real,V <: StaticArray{Tuple{N},T,1},F,G} <: Filter
    μ::V # mean vector
    Σ::SMatrix{N,N,T} # covariance matrix
    Q::SMatrix{N,N,Float64} # process noise
    R::SMatrix{P,P,Float64} # measurement noise
    λ::Float64
    n::Int
    transition_model::F
    observation_model::G
end
const UKF = UnscentedKalmanFilter
function UnscentedKalmanFilter(
    μ::StaticArray,Σ::MatrixLike,Q::MatrixLike,R::MatrixLike,λ,n,sysF,sysG)
    N = size(μ,1)
    P = size(R,1)
    UKF(μ,SMatrix{N,N}(Σ),SMatrix{N,N}(Q),SMatrix{P,P}(R),float(λ),n,sysF,sysG)
end
function UnscentedKalmanFilter(
    μ::MatrixLike,Σ::MatrixLike,Q::MatrixLike,R::MatrixLike,λ,n,sysF,sysG)
    N = size(μ,1)
    P = size(R,1)
    UKF(SVector{N}(μ),SMatrix{N,N}(Σ),SMatrix{N,N}(Q),SMatrix{P,P}(R),float(λ),n,sysF,sysG)
end
function unscented_transform(μ,Σ,λ,n)
    # μv = SVector(μ) # convert to SVector for case of
    Δ = hcat([zeros(length(μ)),sqrt((n+λ)*Σ),-sqrt((n+λ)*Σ)]...)
    # σ_pts = hcat([μv, μv.+sqrt(Σ), μv.-sqrt(Σ)]...)
    σ_pts = [μ + Δ[:,i] for i in 1:size(Δ,2)]
    weights = ones(2*n+1) * 1.0 / (2*(n + λ))
    weights[1] = λ / (n + λ)
    # weights = weights / sum(weights)
    return σ_pts, weights
end
function inverse_unscented_transform(σ_pts,weights)
    # μ = σ_pts * weights
    μ = sum(σ_pts .* weights)
    # Δ = σ_pts .- μ
    # Σ = Δ * diagm((0=>weights)) * Δ'
    Σ = sum([w*(pt-μ)*(pt-μ)' for (pt,w) in zip(σ_pts,weights)])

    return μ,Σ
end
function predict(m::UKF,μ::V,Σ,u) where V
    σ_pts, weights = unscented_transform(μ,Σ,m.λ,m.n)
    σ_pts = [propagate(deterministic(m.transition_model),pt,u) for pt in σ_pts]
    (μ,Σ) = inverse_unscented_transform(σ_pts,weights)
    μ = convert(V,μ)
    Σ = Σ + m.Q
    return μ,Σ
end
function predict!(m::UKF,u)
    m.μ,m.Σ = predict(m,m.μ,m.Σ,u)
end
function update(m::UKF,μ::V,Σ::M,z) where {V,M}
    σ_pts, weights = unscented_transform(μ,Σ,m.λ,m.n)
    σ_pts = [Array(p) for p in σ_pts]
    # Z = hcat([observe(deterministic(m.observation_model),pt) for pt in σ_pts]...)
    # ẑ = Z * weights
    Z = hcat([Array(observe(deterministic(m.observation_model),pt)) for pt in σ_pts]...)
    # ẑ = sum(Z .* weights)
    ẑ = Z * weights
    W = diagm((0=>weights))
    Szz = (Z .- z)*W*(Z .- z)' + m.R
    Sxz = (hcat(σ_pts...) .- μ) * W * (Z .- z)'
    Kt = Sxz*inv(Szz)

    μ = convert(V, μ + Kt*(z .- ẑ))
    Σ = convert(M, Σ - Kt*Szz*Kt')

    return μ,Σ
end
function update!(m::UKF,u)
    m.μ,m.Σ = update(m,m.μ,m.Σ,u)
end

mutable struct BinaryDiscreteFilter{N,V,F,G} <: Filter
    pts                 ::V # distribution over possible locations of target
    μ                   ::SVector{N,Float64} # hypothesis
    # kdtree              ::NearestNeighbors.KDTree
    # k                   ::Int
    transition_model    ::F # = BinaryReachabilityTransitionModel
    observation_model   ::G #
end
function BinaryDiscreteFilter(pts,μ::MatrixLike,sysF,sysG)
    BinaryDiscreteFilter(pts,SVector{size(μ,1)}(μ),sysF,sysG)
end
# function BinaryDiscreteFilter(pts::MatrixLike,μ::MatrixLike,kdtree,k,sysF,sysG)
#     N = length(pts)
#     BinaryDiscreteFilter(
#         SVector{size(pts,1)}(pts),SVector{size(μ,1)}(μ),kdtree,k,sysF,sysG)
# end
# function BinaryDiscreteFilter(pts,k=4,transition_model,observation_model)
#     N = size(pts,1)
#     BinaryDiscreteFilter(
#         pts,
#         SVector{N,Bool}(zeros(N)),
#         KDTree(pts),
#         k,
#         transition_model,
#         observation_model
#     )
# end
function predict(m::BinaryDiscreteFilter,μ,u)
    propagate(m.transition_model,μ,u)
end
function predict!(m::BinaryDiscreteFilter,u)
    m.μ = predict(m,m.μ,u)
end
function update(m::BinaryDiscreteFilter,μ,z)
    """
    TODO: Measurement vector z:
        z[i] = {
                -1      if pts[i] is observed to be empty,
                0       if pts[i] not observed,
                1       if pts[i] observed to contain target
                    }
        z is a point estimate of the target location in R²
    """
    # idxs, dists = knn(m.kdtree, z, m.k)
    # N = size(μ,1)
    # SVector{size(μ,1),Int}([(i ∈ idxs) for i in 1:N])
    h = μ .* measurement_likelihood(m.observation_model, μ, z)
    h = h ./ sum(h)
    μ = SVector{size(μ,1),Float64}(h .> 0)
end
function update!(m::BinaryDiscreteFilter,z)
    m.μ = update(m,m.μ,z)
end

# mutable struct HistogramFilter{N,V,F,G} <: Filter
#     pts                 ::SVector{N,V} # distribution over possible locations of target
#     μ                   ::SVector{N,Bool} # hypothesis
#     kdtree              ::NearestNeighbors.KDTree
#     n                   ::Int
#     transition_model    ::BinaryReachabilityTransitionModel
#     observation_model   ::G
# end
# mutable struct MultiHypothesisFilter{F} <: Filter
#     weights::Vector{Float64}
#     filters::Vector{F}
# end
# const MHF = MultiHypothesisFilter

"""
    A filter model that only responds to the predict!() call every Δt seconds
"""
mutable struct TimedFilter{F<:Filter} <: Filter
    filter::F
    Δt::Float64
    t::Float64
end
function predict(m::TimedFilter,μ,t::Float64)
    if t - m.t >= m.Δt
        return predict(m.filter,μ)
    end
    return μ
end
function predict(m::TimedFilter,μ,u,t::Float64)
    if t - m.t >= m.Δt
        return predict(m.filter,μ,u)
    end
    return μ
end
function predict!(m::TimedFilter,u,t::Float64)
    if t - m.t >= m.Δt
        predict!(m.filter,u)
        m.t = t
    end
    m
end
function predict!(m::TimedFilter,t::Float64)
    if t - m.t >= m.Δt
        predict!(m.filter)
        m.t = t
    end
    m
end
update!(m::TimedFilter,z) = update!(m.filter,z)


include("filters/quad_tree_filter.jl")


end # module
