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

"""
    deterministic(m::T where T <: ObservationModel)

    returns a deterministic version of the sensor model
"""
deterministic(m::DeterministicObservationModel) = m
# deterministic(m::ProbabilisticObservationModel) = throw(MethodError(deterministic, string("deterministic(m::",typeof(m),") not defined")))

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

struct RangeSensor{N,T} <: ObservationModel
    x::SVector{N,T} # sensor origin
end
RangeSensor(x::Union{Vector,Real}) = RangeSensor(SVector{size(x,1)}(x))
input_size(m::RangeSensor) = size(m.x,1)
observe(m::RangeSensor, x) = norm(x - m.x)
observe(m::RangeSensor, x, u) = observe(m, x)
measurement_jacobian(m::RangeSensor, x) = (x - m.x) / norm(x - m.x)

struct GaussianRangeSensor{N} <: ObservationModel
    x::SVector{N,Float64} # sensor origin
    R::SMatrix{N,N,Float64}
    m_noise::MultivariateNormal
end
deterministic(m::GaussianRangeSensor) = RangeSensor(m.x)
function GaussianRangeSensor(x::SMatrix,R::SMatrix)
    GaussianRangeSensor(x,R,MultivariateNormal(zeros(size(x,1)),Matrix(R)))
end
function GaussianRangeSensor(x::MatrixLike,R::MatrixLike)
    N = size(x,1)
    GaussianRangeSensor(SVector{N}(x),SMatrix{N,N}(R),MultivariateNormal(zeros(N),R))
end
input_size(m::GaussianRangeSensor) = size(m.x,1)
observe(m::GaussianRangeSensor, x) = norm(x - m.x + rand(m.m_noise))
observe(m::GaussianRangeSensor, x, u) = observe(m, x)
measurement_jacobian(m::GaussianRangeSensor, x) = (x - m.x) / norm(x - m.x)

struct BearingSensor{N}
    x::SVector{N,Float64}
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


# composite sensor models
struct CompositeSensorModel{T}
    models::T
    input_size::Int
end

observe(cm::CompositeSensorModel{T} where {T <: Tuple},x) = vcat([observe(m,x) for m in cm.models]...)
measurement_jacobian(cm::CompositeSensorModel{T} where {T <: Tuple},x) = vcat([measurement_jacobian(m,x) for m in model]...)
