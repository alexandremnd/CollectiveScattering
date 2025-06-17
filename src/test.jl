using CUDA, GPUArrays
using Polyester
using BenchmarkTools

include("structs.jl")
include("utils.jl")
include("lattice.jl")
include("field.jl")
include("computation.jl")
include("averaging.jl")
include("misc.jl")

function check_matrix(m)
    reference_m = [-0.5+0.0im 0.0456522-0.00517052im -0.00513767-0.0223902im -0.0144465+0.00508321im;
   0.0456522-0.00517052im         -0.5+0.0im           0.0456522-0.00517052im  -0.00513767-0.0223902im;
 -0.00513767-0.0223902im     0.0456522-0.00517052im         -0.5+0.0im           0.0456522-0.00517052im;
  -0.0144465+0.00508321im  -0.00513767-0.0223902im     0.0456522-0.00517052im         -0.5+0.0im]

    if isapprox(m, reference_m; atol=1e-3)
        println("Matrix check passed!")
    else
        println("Matrix check failed!")
    end
    println("Expected:")
    display(reference_m)
    println("Got:")
    display(m)
end

function check_field(e)
    reference_e = [0.0 + 0.0005im; -7.678812992292257e-6 + 0.00044111614469105945im; -6.036044453229109e-6 + 0.00030320506501231267im; 1.1864244780935834e-6 + 0.00016268189705423308im]

    if isapprox(e, reference_e; atol=1e-7)
        println("Field check passed!")
    else
        println("Field check failed!")
    end
    println("Expected:")
    display(reference_e)
    println("Got:")
    display(e)
end

function test_code()
    Na  = 4
    Nd  = 20
    Rd  = 9.0
    a   = 0.07
    Δ0  = 0.0

    # Electric field parameters
    E0  = 1e-3
    w0  = 4.0
    θ   = deg2rad(90)
    d   = bragg_periodicity(deg2rad(10))

    incident_field = GaussianBeam(E0, w0, θ)
    params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)

    s = [0.0 0 0; 1 1 1; 2 2 2; 3 3 3]
    m = zeros(ComplexF64, 4, 4)
    e = zeros(ComplexF64, 4)
    a = zeros(ComplexF64, 4)

    compute_system_matrix!(m, s, Δ0)
    e = compute_field!(e, s[:, 1], s[:, 2], s[:, 3], incident_field)
    e .*= 0.5im

    check_matrix(m)
    check_field(e)
end

test_code()