let
    dt = 0.1
    single_integrator_1D(dt)
    single_integrator_2D(dt)
    blended_single_integrator_2D(dt)
    double_integrator_1D(dt)
    double_integrator_2D(dt)
end
let
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
end
