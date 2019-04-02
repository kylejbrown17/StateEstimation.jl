let
    #### Observation Models
    C = Matrix{Float64}(I,2,2)
    D = fill!(Matrix{Float64}(undef,2,1),0.0)
    m = LinearSensor(C,D)
    input_size(m)
    x = [1.0;1.0]
    u = [1.0]
    measurement_jacobian(m,observe(m,x))
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
    m = BoundedSensor(m,5.0)
    @test any(isnan, observe(m,[10.0;10.0],rand(2)))
    @test any(isnan, observe(m,[10.0;10.0]))
    @test measurement_jacobian(m,x) == measurement_jacobian(m.sensor,x)

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
    m = BoundedSensor(m,5.0)
    @test any(isnan, observe(m,[10.0;10.0],rand(2)))
    @test any(isnan, observe(m,[10.0;10.0]))
    @test measurement_jacobian(m,x) == measurement_jacobian(m.sensor,x)

    m = GaussianBearingSensor([0.0;0.0],R)
    input_size(m)
    measurement_jacobian(m,observe(m,x))
    @test (x - m.x)/norm(x) != observe(m,x)
    @test (x - m.x)/norm(x) != observe(m,x,u)
    @test abs(norm(observe(m,x)) - 1.0) < 0.0000001

    m = CompositeSensorModel((RangeSensor([0.0;0.0]),BearingSensor([0.0;0.0])))
end
