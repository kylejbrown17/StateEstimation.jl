using Test
using StateEstimation
using LinearAlgebra

x = [0.0;1.0]
A = [1.0 0.1; 0.0 1.0]
B = reshape([0.0; 0.1],(2,1))
sys = DiscreteLinearSystem(x,A,B)
@test ([0.1;1.0] == propagate!(sys,[0.0]))
@test state_jacobian(sys) == sys.A

Q = [1.0 0.0; 0.0 1.0]
sys = DiscreteLinearGaussianSystem(x,A,B,Q)
@test ([0.1;1.0] != propagate!(sys,[0.0]))
@test state_jacobian(sys) == sys.A

#### Observation Models
C = Matrix{Float64}(I,2,2)
D = fill!(Matrix{Float64}(undef,2,1),0.0)
m = LinearSensor(C,D)
x = [1.0;1.0]
u = [1.0]
@test x == observe(m,x)
@test x == observe(m,x,u)

R = Matrix{Float64}(I,2,2)
m = LinearGaussianSensor(C,D,R)
@test x != observe(m,x)
@test x != observe(m,x,u)

m = RangeSensor([0.0;0.0])
@test norm(x - m.x) == observe(m,x)
@test norm(x - m.x) == observe(m,x,u)

m = GaussianRangeSensor([0.0;0.0],R)
@test norm(x - m.x) != observe(m,x)
@test norm(x - m.x) != observe(m,x,u)

m = BearingSensor([0.0;0.0])
@test (x - m.x)/norm(x) == observe(m,x)
@test (x - m.x)/norm(x) == observe(m,x,u)

m = GaussianBearingSensor([0.0;0.0],R)
@test (x - m.x)/norm(x) != observe(m,x)
@test (x - m.x)/norm(x) != observe(m,x,u)
@test norm(observe(m,x)) == 1.0
