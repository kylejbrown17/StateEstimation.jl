export
    ObservationModel,
    DeterministicObservationModel,
    ProbabilisticObservationModel,

    MobileSensor,
    LinearSensor,
    identity_sensor,
    LinearGaussianSensor,
    RangeSensor,
    GaussianRangeSensor,
    BearingSensor,
    GaussianBearingSensor,
    AdditiveGaussianSensor,
    BoundedSensor,
    CompositeSensorModel,
    observe,
    in_range,
    input_size,
    measurement_jacobian

# abstract type ObservationModel end
abstract type DeterministicObservationModel <: ObservationModel end
abstract type ProbabilisticObservationModel <: ObservationModel end
abstract type DiscreteObservationModel <: ObservationModel end

"""
    deterministic(m::T where T <: ObservationModel)

    returns a deterministic version of the sensor model
"""
deterministic(m::DeterministicObservationModel) = m
get_range(m::ObservationModel) = Inf
in_range(m::ObservationModel,x) = true

mutable struct MobileSensor{S,T<:Transformation}
    sensor::S
    transform::T # transform from sensor frame to global frame
end
input_size(m::MobileSensor) = input_size(m.sensor)

## Observation Models
struct LinearSensor{N,M,P,T} <: DeterministicObservationModel
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
input_size(m::LinearSensor) = size(m.C,2)
observe(m::LinearSensor,x) = m.C*x
observe(m::LinearSensor,x,u) = m.C*x + m.D*u
measurement_jacobian(m::LinearSensor, x) = m.C

mutable struct RangeSensor{N,T} <: DeterministicObservationModel
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

mutable struct GaussianRangeSensor{N} <: ProbabilisticObservationModel
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

mutable struct BearingSensor <: DeterministicObservationModel
    x::SVector{2,Float64}
end
BearingSensor(x::MatrixLike) = BearingSensor(SVector{size(x,1)}(x))
input_size(m::BearingSensor) = 2
observe(m::BearingSensor, x) = (x - m.x) / norm(x - m.x)
observe(m::BearingSensor, x, u) = observe(m, x)
# measurement_jacobian(m::BearingSensor, x) = ([0.0 1.0;-1.0 0.0]*x)*([0.0 1.0;-1.0 0.0]*x)' * norm(x)^-3
measurement_jacobian(m::BearingSensor, x) = SMatrix{2,2}([x[2]^2 -x[1]*x[2]; -x[2]*x[1] x[1]^2]) / (norm(x)^3)

mutable struct GaussianBearingSensor <: ProbabilisticObservationModel
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

# Observation model that returns all occupied and unoccupied nodes of a graph
# struct DiscreteGraphSensor{N,V}
#     """
#     all nodes in neighbors(G,x) are observable to DiscreteGraphSensor
#     """
#     # lg_sensor           ::LinearGaussianSensor{N,}
#     # pts                 ::SVector{N,V} # distribution over possible locations of target
#     # kdtree              ::NearestNeighbors.KDTree
#     # k                   ::Int           # number of nearest neighbors to hit
#     G                   ::MetaGraph
#     # x                   ::Int
# end
# function observe(m:::DiscreteGraphSensor, x)
#     idxs, dists = get_neighbors(m.kdtree, x, m.k)
#     hits = intersect(neighbors(m.G,m.x),idxs) # return intersection of visible nodes and nodes in x
#
# end
# observe(m::DiscreteGraphSensor, x, u) = observe(m, x)
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


struct BoundedSensor{S,B}
    sensor::S
    bounds::B
end
in_range(m::BoundedSensor, x) = true
function observe(m::BoundedSensor, x, u)
    z = observe(m.sensor, x, u)
    if in_range(m, x)
        return z
    else
        return z*NaN # return an invalid version of the elements
    end
end
function observe(m::BoundedSensor, x)
    z = observe(m.sensor, x)
    if in_range(m, x)
        return z
    else
        return z*NaN # return an invalid version of the elements
    end
end
# in_range(m::BoundedSensor{S,B} where {S<:LinearSensor,B}, x) = norm(z) < m.bounds
in_range(m::BoundedSensor{S,B} where {S<:RangeSensor,B}, x) = norm(observe(m.sensor,x)) < m.bounds
in_range(m::BoundedSensor{S,B} where {S<:BearingSensor,B}, x) = norm(m.sensor.x-x) < m.bounds
in_range(m::BoundedSensor{S,B} where {S<:LinearSensor,B}, x) = norm(m.sensor.x-x) < m.bounds
function measurement_jacobian(m::BoundedSensor, x)
    z = observe(m.sensor, x)
    ∇ = measurement_jacobian(m.sensor, x)
    if in_range(m, z)
        return ∇
    else
        return 0 * ∇
    end
end
deterministic(m::BoundedSensor) = BoundedSensor(deterministic(m.sensor),m.bounds)
input_size(m::BoundedSensor) = input_size(m.sensor)

struct AdditiveGaussianSensor{S,N} <: ProbabilisticObservationModel
    sensor::S
    R::SMatrix{N,N,Float64}
    m_noise::MultivariateNormal
end
observe(m::AdditiveGaussianSensor,x,u) = observe(m.sensor,x,u) + rand(m.m_noise)
observe(m::AdditiveGaussianSensor,x) = observe(m.sensor,x) + rand(m.m_noise)
measurement_jacobian(m::AdditiveGaussianSensor,x) = measurement_jacobian(m.sensor,x)
deterministic(m::AdditiveGaussianSensor) = deterministic(m.sensor)
input_size(m::AdditiveGaussianSensor) = input_size(m.sensor)

observe(m::AdditiveGaussianSensor{S,N} where {S<:RangeSensor,N},x,u) = observe(m.sensor,x+rand(m.m_noise),u)
observe(m::AdditiveGaussianSensor{S,N} where {S<:RangeSensor,N},x) = observe(m.sensor,x+rand(m.m_noise))

const LinearGaussianSensor{N,M,P,T} = AdditiveGaussianSensor{LinearSensor{N,M,P,T},N}
LinearGaussianSensor(C,D,R) = AdditiveGaussianSensor(LinearSensor(C,D),SMatrix{size(R)...}(R),MultivariateNormal(zeros(size(C,1)),Matrix(R)))
LinearGaussianSensor(sys::LinearSensor,R) = AdditiveGaussianSensor(sys,SMatrix{size(R)...}(R),MultivariateNormal(zeros(size(sys.C,1)),Matrix(R)))


# composite sensor models
struct CompositeSensorModel{T}
    models::T
    input_size::Int
end
function CompositeSensorModel(models::T where {T <: Tuple})
    for m in models
        @assert input_size(m) == input_size(models[1])
    end
    CompositeSensorModel(models,input_size(models[1]))
end
deterministic(m::CompositeSensorModel) = CompositeSensorModel(tuple([deterministic(s for s in m.models)]...),m.input_size)
observe(m::CompositeSensorModel{T} where T,x,u) = vcat([observe(s,x,u) for s in m.models])
observe(m::CompositeSensorModel{T} where T,x) = vcat([observe(s,x) for s in m.models])
measurement_jacobian(m::CompositeSensorModel{T} where T,x) = vcat([measurement_jacobian(s,x) for s in m.models])
input_size(m::CompositeSensorModel) = m.input_size

observe(cm::CompositeSensorModel{T} where {T <: Tuple},x) = vcat([observe(m,x) for m in cm.models]...)
measurement_jacobian(cm::CompositeSensorModel{T} where {T <: Tuple},x) = vcat([measurement_jacobian(m,x) for m in model]...)
