export
    ObservationModel,
    DeterministicObservationModel,
    ProbabilisticObservationModel,

    LinearSensor,
    LinearGaussianSensor,
    RangeSensor,
    GaussianRangeSensor,
    BearingSensor,
    GaussianBearingSensor,
    predict_obs,
    observe,
    input_size,
    measurement_jacobian

abstract type ObservationModel end
abstract type DeterministicObservationModel <: ObservationModel end
abstract type ProbabilisticObservationModel <: ObservationModel end
abstract type DiscreteObservationModel <: ObservationModel end

"""
    deterministic(m::T where T <: ObservationModel)

    returns a deterministic version of the sensor model
"""
deterministic(m::DeterministicObservationModel) = m
# deterministic(m::ProbabilisticObservationModel) = throw(MethodError(deterministic, string("deterministic(m::",typeof(m),") not defined")))
get_range(m::ObservationModel) = Inf
in_range(m::ObservationModel,x) = true

## Observation Models
struct LinearSensor{N,M,P,T} <: ObservationModel
    C::SMatrix{P,N,T}
    D::SMatrix{P,M,T}
end
function LinearSensor(C::MatrixLike,D::MatrixLike)
    p = size(C,1)
    n = size(C,2)
    m = size(D,2)
    LinearSensor(SMatrix{p,n}(C),SMatrix{p,m}(D))
end
function identity_sensor(n::Int,m::Int,T::DataType=Float64)
    LinearSensor(Matrix{T}(I,n,n),zeros(T,(n,m)))
end
function identity_sensor(x::Vector,u::Vector)
    identity_sensor(size(x,1),size(u,1),eltype(x))
end
# function LinearSensor(C::Matrix,D::Vector)
#     p = size(C,1)
#     n = size(C,2)
#     m = 1
#     LinearSensor(SMatrix{p,n}(C),SMatrix{p,m}(D))
# end
input_size(m::LinearSensor) = size(m.C,2)
observe(m::LinearSensor,x) = m.C*x
observe(m::LinearSensor,x,u) = m.C*x + m.D*u
measurement_jacobian(m::LinearSensor, x) = m.C

struct LinearGaussianSensor{N,M,P,T} <: ObservationModel
    C::SMatrix{P,N,T}
    D::SMatrix{P,M,T}
    R::SMatrix{P,P,Float64}
    m_noise::MultivariateNormal
end
LinearGaussianSensor(C::SMatrix,D::SMatrix,R::SMatrix) = LinearGaussianSensor(C,D,R,MultivariateNormal(zeros(size(C,1)),Matrix(R)))
function LinearGaussianSensor(C::MatrixLike,D::MatrixLike,R::MatrixLike)
    p = size(C,1)
    n = size(C,2)
    m = size(D,2)
    LinearGaussianSensor(SMatrix{p,n}(C),SMatrix{p,m}(D),SMatrix{p,p}(R))
end
deterministic(m::LinearGaussianSensor) = LinearSensor(m.C,m.D)
input_size(m::LinearGaussianSensor) = size(m.C,2)
observe(m::LinearGaussianSensor,x) = m.C*x + rand(m.m_noise)
observe(m::LinearGaussianSensor,x,u) = m.C*x + m.D*u + rand(m.m_noise)
measurement_jacobian(m::LinearGaussianSensor, x) = m.C

# struct RangeSensor{N,T} <: ObservationModel
#     x::SVector{N,T} # sensor origin
# end
# RangeSensor(x::Union{Vector,Real}) = RangeSensor(SVector{size(x,1)}(x))
# input_size(m::RangeSensor) = size(m.x,1)
# observe(m::RangeSensor, x) = norm(x - m.x)
# observe(m::RangeSensor, x, u) = observe(m, x)
# measurement_jacobian(m::RangeSensor, x) = (x - m.x) / norm(x - m.x)

struct RangeSensor{N,T} <: ObservationModel
    x::SVector{N,T}     # sensor origin
    r::Float64          # maximum sensor range
end
RangeSensor(x::Union{Vector,Real}) = RangeSensor(SVector{size(x,1)}(x),Inf)
input_size(m::RangeSensor) = size(m.x,1)
in_range(m::RangeSensor,x) = (norm(x-m.x) < m.r)
get_range(m::RangeSensor) = m.r
function observe(m::RangeSensor, x)
    d = norm(x - m.x)
    d < m.r ? d : NaN
end
observe(m::RangeSensor, x, u) = observe(m, x)
function measurement_jacobian(m::RangeSensor, x)
    Δ = x - m.x
    if in_range(m,x)
        return Δ / norm(Δ)
    else
        return 0 * Δ
    end
end

struct GaussianRangeSensor{N} <: ObservationModel
    x::SVector{N,Float64} # sensor origin
    R::SMatrix{N,N,Float64}
    m_noise::MultivariateNormal
    r::Float64          # maximum sensor range
end
deterministic(m::GaussianRangeSensor) = RangeSensor(m.x,m.r)
function GaussianRangeSensor(x::SVector,R::SMatrix,r::Float64)
    GaussianRangeSensor(x,R,MultivariateNormal(zeros(size(x,1)),Matrix(R)),r)
end
function GaussianRangeSensor(x::SVector,R::SMatrix)
    GaussianRangeSensor(x,R,Inf)
end
function GaussianRangeSensor(x::MatrixLike,R::MatrixLike,r::Float64)
    N = size(x,1)
    GaussianRangeSensor(SVector{N}(x),SMatrix{N,N}(R),r)
end
function GaussianRangeSensor(x::MatrixLike,R::MatrixLike)
    N = size(x,1)
    GaussianRangeSensor(SVector{N}(x),SMatrix{N,N}(R))
end
input_size(m::GaussianRangeSensor) = size(m.x,1)
observe(m::GaussianRangeSensor, x) = norm(x - m.x + rand(m.m_noise))
observe(m::GaussianRangeSensor, x, u) = observe(m, x)
measurement_jacobian(m::GaussianRangeSensor, x) = (x - m.x) / norm(x - m.x)

struct BearingSensor
    x::SVector{2,Float64}
end
BearingSensor(x::MatrixLike) = BearingSensor(SVector{size(x,1)}(x))
input_size(m::BearingSensor) = 2
observe(m::BearingSensor, x) = (x - m.x) / norm(x - m.x)
observe(m::BearingSensor, x, u) = observe(m, x)
# measurement_jacobian(m::BearingSensor, x) = ([0.0 1.0;-1.0 0.0]*x)*([0.0 1.0;-1.0 0.0]*x)' * norm(x)^-3
measurement_jacobian(m::BearingSensor, x) = [x[2]^2 -x[1]*x[2]; -x[2]*x[1] x[1]^2] / (norm(x)^3)

struct GaussianBearingSensor
    x::SVector{2,Float64}
    R::SMatrix{2,2,Float64}
    m_noise::MultivariateNormal
end
GaussianBearingSensor(x::MatrixLike) = GaussianBearingSensor(SMatrix{2,2}(x))
deterministic(m::GaussianBearingSensor) = BearingSensor(m.x)
GaussianBearingSensor(x,R) = GaussianBearingSensor(x,R,MultivariateNormal(zeros(size(x)),R))
input_size(m::GaussianBearingSensor) = 2
function observe(m::GaussianBearingSensor, x)
    y = x - m.x + rand(m.m_noise)
    return y / norm(y)
end
observe(m::GaussianBearingSensor, x, u) = observe(m, x)
measurement_jacobian(m::GaussianBearingSensor, x) = SMatrix{2,2}(([0.0 1.0;-1.0 0.0]*x)*([0.0 1.0;-1.0 0.0]*x)') * norm(x)^-3

# mutable struct NodeObservationModel{N,V}
#     # G::MetaGraph        # graph
#     pts::Vector{V}      # vector of points corresponding to graph nodes
#     kdtree::KDTree      # kdtree for lookup
#     x::SVector{N}       # position of sensor
#     r::Float64          # sensor range
#     k::Int              # num neighbors for knn query
# end
# function NodeObservationModel(pts::Vector,x,r::Float64=Inf,k::Int=4)
#     NodeObservationModel(pts,KDTree(hcat(pts...)),x,r,k)
# end
# function set_position!(m::NodeObservationModel,x)
#     m.x = x
# end
# get_range(m::NodeObservationModel) = m.r
# in_range(m::NodeObservationModel,x) = norm(x-m.x) < get_range(m)
# function observe(m::NodeObservationModel,x)
#     z = zeros(Int,length(pts))
#     if in_range(m,x)
#         idxs = inrange(m.kdtree, x)
#         z[idxs] = -1
#         idxs, dists = knn(m.kdtree, x, m.n)
#         z[idxs] = 1
#     end
#     return z
# end
#
# mutable struct GraphObservationModel
#     G::MetaGraph    # Graph
#     D::Matrix       # Distance matrix
#     x::Int          # index of sensor location
# end
# function set_position!(m::GraphObservationModel,x)
#     m.x = x
# end

# composite sensor models
struct CompositeSensorModel{T}
    models::T
    input_size::Int
end

observe(cm::CompositeSensorModel{T} where {T <: Tuple},x) = vcat([observe(m,x) for m in cm.models]...)
measurement_jacobian(cm::CompositeSensorModel{T} where {T <: Tuple},x) = vcat([measurement_jacobian(m,x) for m in model]...)
