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
    observe,
    measurement_jacobian

abstract type ObservationModel end
abstract type DeterministicObservationModel <: ObservationModel end
abstract type ProbabilisticObservationModel <: ObservationModel end

## Observation Models
struct LinearSensor <: ObservationModel
    C::Matrix{Float64}
    D::Matrix{Float64}
end
input_size(m::LinearSensor) = size(m.C,2)
observe(m::LinearSensor,x) = m.C*x

observe(m::LinearSensor,x,u) = m.C*x + m.D*u
measurement_jacobian(m::LinearSensor, x) = m.C
struct LinearGaussianSensor <: ObservationModel
    C::Matrix{Float64}
    D::Matrix{Float64}
    R::Matrix{Float64}
    m_noise::MultivariateNormal
end
LinearGaussianSensor(C,D,R) = LinearGaussianSensor(C,D,R,MultivariateNormal(zeros(size(C,1)),R))
input_size(m::LinearGaussianSensor) = size(m.C,2)
observe(m::LinearGaussianSensor,x) = m.C*x + rand(m.m_noise)
observe(m::LinearGaussianSensor,x,u) = m.C*x + m.D*u + rand(m.m_noise)
measurement_jacobian(m::LinearGaussianSensor, x) = m.C + rand(m.m_noise)
struct RangeSensor <: ObservationModel
    x::Vector{Float64} # sensor origin
end
input_size(m::RangeSensor) = size(m.x,1)
observe(m::RangeSensor, x) = norm(x - m.x)
observe(m::RangeSensor, x, u) = norm(x - m.x)
measurement_jacobian(m::RangeSensor, x) = (x - m.x) / norm(x - m.x)
struct GaussianRangeSensor <: ObservationModel
    x::Vector{Float64} # sensor origin
    R::Matrix{Float64}
    m_noise::MultivariateNormal
end
GaussianRangeSensor(x,R) = GaussianRangeSensor(x,R,MultivariateNormal(zeros(size(x)),R))
input_size(m::GaussianRangeSensor) = size(m.x,1)
observe(m::GaussianRangeSensor, x) = norm(x - m.x + rand(m.m_noise))
observe(m::GaussianRangeSensor, x, u) = norm(x - m.x + rand(m.m_noise))
measurement_jacobian(m::GaussianRangeSensor, x) = (x - m.x) / norm(x - m.x)
struct BearingSensor
    x::Vector{Float64}
end
input_size(m::BearingSensor) = 2
observe(m::BearingSensor, x) = (x - m.x) / norm(x - m.x)
observe(m::BearingSensor, x, u) = (x - m.x) / norm(x - m.x)
# measurement_jacobian(m::BearingSensor, x) = ([0.0 1.0;-1.0 0.0]*x)*([0.0 1.0;-1.0 0.0]*x)' * norm(x)^-3
measurement_jacobian(m::BearingSensor, x) = [x[2]^2 -x[1]*x[2]; -x[2]*x[1] x[1]^2] / (norm(x)^3)
struct GaussianBearingSensor
    x::Vector{Float64}
    R::Matrix{Float64}
    m_noise::MultivariateNormal
end
GaussianBearingSensor(x,R) = GaussianBearingSensor(x,R,MultivariateNormal(zeros(size(x)),R))
input_size(m::GaussianBearingSensor) = 2
function observe(m::GaussianBearingSensor, x)
    y = x - m.x + rand(m.m_noise)
    return y / norm(y)
end
function observe(m::GaussianBearingSensor, x, u)
    y = x - m.x + rand(m.m_noise)
    return y / norm(y)
end
measurement_jacobian(m::GaussianBearingSensor, x) = ([0.0 1.0;-1.0 0.0]*x)*([0.0 1.0;-1.0 0.0]*x)' * norm(x)^-3
observe(model::T where {T <: Tuple},x) = vcat([observe(m,x) for m in model]...)
measurement_jacobian(model::T where {T <: Tuple},x) = vcat([measurement_jacobian(m,x) for m in model]...)
