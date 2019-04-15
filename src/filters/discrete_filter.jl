export
    DiscreteFilter,
    predict,
    predict!,
    update,
    update!

mutable struct DiscreteFilter{N,V,F,G} <: Filter
    pts                 ::V # distribution over possible locations of target
    μ                   ::SVector{N,Float64} # hypothesis
    transition_model    ::F # = BinaryReachabilityTransitionModel
    observation_model   ::G #
end
function DiscreteFilter(pts,μ::MatrixLike,sysF,sysG)
    DiscreteFilter(pts,SVector{size(μ,1)}(μ),sysF,sysG)
end
function predict(m::DiscreteFilter,μ,u)
    propagate(m.transition_model,μ,u)
end
function predict!(m::DiscreteFilter,u)
    m.μ = predict(m,m.μ,u)
end
function update(m::DiscreteFilter,μ,z)
    """
    TODO: Measurement vector z:
        z[i] = {
                -1      if pts[i] is observed to be empty,
                0       if pts[i] not observed,
                1       if pts[i] observed to contain target
                    }
        z is a point estimate of the target location in R²
    """
    h = μ .* measurement_likelihood(m.observation_model, μ, z)
    μ = h ./ sum(h)
end
function update!(m::DiscreteFilter,z)
    m.μ = update(m,m.μ,z)
end
