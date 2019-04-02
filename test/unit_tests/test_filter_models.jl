
# Filters
let
    n = 2; m = 1; p = 2
    x = [0.0;1.0]
    u = [1.0]
    μ₀ = [0.0;1.0]
    Σ₀ = Matrix{Float64}(I,2,2)
    A = [1.0 0.1; 0.0 1.0]
    B = reshape([0.0; 0.1],(2,1))
    C = Matrix{Float64}(I,2,2)
    D = fill!(Matrix{Float64}(undef,2,1),0.0)
    Q = Matrix{Float64}(I,2,2)
    R = Matrix{Float64}(I,2,2)
    sysF = DiscreteLinearGaussianSystem(A,B,Q)
    sysG = LinearGaussianSensor(C,D,R)

    m = KalmanFilter(VecE2(μ₀),Σ₀,sysF,sysG)
    @test typeof(predict(m,m.μ,m.Σ,u)[1]) <: VecE2
    update(m,m.μ,m.Σ,observe(deterministic(m.observation_model),x))
    predict!(m,u)
    @test typeof(m.μ) <: VecE2
    update!(m,observe(m.observation_model,x))

    m = KalmanFilter(μ₀,Σ₀,sysF,sysG)
    m = KalmanFilter(μ₀,Σ₀,A,B,C,D,Q,R)
    predict(m,m.μ,m.Σ,u)
    update(m,m.μ,m.Σ,observe(deterministic(m.observation_model),x))
    predict!(m,u)
    update!(m,observe(m.observation_model,x))

    m = EKF(μ₀,Σ₀,Q,R,sysF,sysG)
    predict(m,m.μ,m.Σ,u)
    update(m,m.μ,m.Σ,observe(deterministic(m.observation_model),x))
    predict!(m,u)
    update!(m,observe(m.observation_model,x))

    m = EKF(VecE2(μ₀),Σ₀,Q,R,sysF,sysG)
    @test typeof(predict(m,m.μ,m.Σ,u)[1]) <: VecE2
    update(m,m.μ,m.Σ,observe(deterministic(m.observation_model),x))
    predict!(m,u)
    @test typeof(m.μ) <: VecE2
    update!(m,observe(m.observation_model,x))

    m = UKF(μ₀,Σ₀,Q,R,2,2,sysF,sysG)
    σ_pts, weights = unscented_transform(m.μ,m.Σ,m.λ,m.n)
    inverse_unscented_transform(σ_pts,weights)
    predict(m,m.μ,m.Σ,u)
    update(m,m.μ,m.Σ,observe(deterministic(m.observation_model),x))
    predict!(m,u)
    update!(m,observe(m.observation_model,x))


    m = UKF(VecE2(μ₀),Σ₀,Q,R,2,2,sysF,sysG)
    σ_pts, weights = unscented_transform(m.μ,m.Σ,m.λ,m.n)
    (μᵖ,Σᵖ) = inverse_unscented_transform(σ_pts,weights)
    @test norm(m.μ - μᵖ) < 0.000000001
    @test norm(m.Σ - Σᵖ) < 0.000000001
    @test typeof(predict(m,m.μ,m.Σ,u)[1]) <: VecE2
    update(m,m.μ,m.Σ,observe(deterministic(m.observation_model),x))
    predict!(m,u)
    @test typeof(m.μ) <: VecE2
    update!(m,observe(m.observation_model,x))
end
