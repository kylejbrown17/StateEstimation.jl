using Test
using StateEstimation
using LinearAlgebra
using StaticArrays
using Vec

# function StaticArrays.similar_type(::Type{V}, ::Type{F}, size::Size{N}) where {V<:VecE, F<:AbstractFloat, N <: Tuple}
#     if size == Size(V) && eltype(V) == F
#         return V
#     else # convert back to SArray
#         return SArray{N,F,length(size),prod(size)}
#     end
# end

# Transtion Models
x = [0.0;1.0]
A = [1.0 0.1; 0.0 1.0]
B = [0.0; 0.1]
sys = DiscreteLinearSystem(A,B)
state_jacobian(sys,x)
@test ([0.1;1.0] == StateEstimation.propagate(sys,x,[0.0]))
@test state_jacobian(sys) == sys.A

Q = [1.0 0.0; 0.0 1.0]
sys = DiscreteLinearGaussianSystem(A,B,Q)
state_jacobian(sys,x)
@test ([0.1;1.0] != StateEstimation.propagate(sys,x,[0.0]))
@test state_jacobian(sys) == sys.A

#### Observation Models
C = Matrix{Float64}(I,2,2)
D = fill!(Matrix{Float64}(undef,2,1),0.0)
m = LinearSensor(C,D)
input_size(m)
measurement_jacobian(m,observe(m,x))
x = [1.0;1.0]
u = [1.0]
@test x == observe(m,x)
@test x == observe(m,x,u)

R = Matrix{Float64}(I,2,2)
m = LinearGaussianSensor(C,D,R)
input_size(m)
measurement_jacobian(m,observe(m,x))
@test x != observe(m,x)
@test x != observe(m,x,u)

m = RangeSensor([0.0;0.0])
input_size(m)
measurement_jacobian(m,observe(m,x))
@test norm(x - m.x) == observe(m,x)
@test norm(x - m.x) == observe(m,x,u)

m = GaussianRangeSensor([0.0;0.0],R)
input_size(m)
measurement_jacobian(m,observe(m,x))
@test norm(x - m.x) != observe(m,x)
@test norm(x - m.x) != observe(m,x,u)

m = BearingSensor([0.0;0.0])
input_size(m)
measurement_jacobian(m,observe(m,x))
@test (x - m.x)/norm(x) == observe(m,x)
@test (x - m.x)/norm(x) == observe(m,x,u)

m = GaussianBearingSensor([0.0;0.0],R)
input_size(m)
measurement_jacobian(m,observe(m,x))
@test (x - m.x)/norm(x) != observe(m,x)
@test (x - m.x)/norm(x) != observe(m,x,u)
@test abs(norm(observe(m,x)) - 1.0) < 0.0000001

# Filters
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
inverse_unscented_transform(σ_pts,weights)
@test typeof(predict(m,m.μ,m.Σ,u)[1]) <: VecE2
update(m,m.μ,m.Σ,observe(deterministic(m.observation_model),x))
predict!(m,u)
@test typeof(m.μ) <: VecE2
update!(m,observe(m.observation_model,x))

nx = 4
ny = 4
X = zeros(nx*ny)
X[1] = 1
pts = []
for i in 1:nx
    for j in 1:ny
        push!(pts,[i;j])
    end
end
idxs = pts
A = zeros(Bool,length(pts),length(pts))
for idx in 1:length(x)
    i = div(idx-1,nx) + 1
    j = mod(idx-1,nx) + 1
    if (i,j) in idxs
        for di in [-1,1]
            if (i+di,j) in idxs
                idx2 = (i+di-1)*nx + j
                A[idx,idx2] = true
            end
        end
        for dj in [-1,1]
            if (i,j+dj) in idxs
                idx2 = (i-1)*nx + j+dj
                A[idx,idx2] = true
            end
        end
    end
end
