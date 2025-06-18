using CUDA, GPUArrays
using Polyester
import LinearAlgebra.BLAS: get_num_threads, set_num_threads
using BenchmarkTools

function mean_intensity(params::SimulationParameters, incident_field::GaussianBeam, Δ0_span, iterations=100)
    R_file = open("data/DetuningSearch/reflection-$(SLURM_JOB_ID).csv", "w")

    θ_span = range(deg2rad(incident_field.θ - 1), deg2rad(incident_field.θ + 1), length=5)
    ϕ_span = range(0, π, length=2)
    X, Y, Z = build_sphere_region(45.0, θ_span, ϕ_span)

    R = zeros(length(Δ0_span))

    for i in axes(Δ0_span, 1)
        new_params = SimulationParameters(params.Na, params.Nd, params.Rd, params.a, params.d, Δ0_span[i])
        intensity = compute_mean_intensity(new_params, incident_field, X, Y, Z, iterations)

        intensity_cpu = sum(Array(intensity), dims=1)
        R[i] = intensity_cpu[2] / intensity_cpu[1]
    end

    save_matrix(R_file, R)
    close(R_file)
end

include("structs.jl")
include("utils.jl")
include("lattice.jl")
include("field.jl")
include("computation.jl")
include("averaging.jl")
include("misc.jl")

const SLURM_JOB_ID::String = get(ENV, "SLURM_JOBID", "local")

println("========== Configuration ==========")
set_num_threads(1)
println("Slurm JOBID: $(SLURM_JOB_ID)")
println(get_num_threads(), " threads used for BLAS operations.")
println(Threads.nthreads(), " threads used for Julia operations.")
# pin_thread()
println("====================================")

intensity_file = open("data/Dynamic/intensity-$(SLURM_JOB_ID).csv", "w")
parameters_file = open("data/Dynamic/parameters-$(SLURM_JOB_ID).txt", "w")
time_file = open("data/Dynamic/time-$(SLURM_JOB_ID).txt", "w")

# Simulation parameters
Na  = 600
Nd  = 40
Rd  = 9.0
a   = 0.07
Δ0  = 0.25

# Electric field parameters
E0  = 1e-3
w0  = 4.0
θ   = deg2rad(90)
d   = bragg_periodicity(deg2rad(10))

incident_field = GaussianBeam(E0, w0, θ)
params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)


mean_intensity(params, incident_field, range(-3.0, 3.0, length=100), 5000)


# =========== Dynamic intensity computation ===========
t_span = 0:0.05:1.0  # Time span for dynamic intensity calculation
θ_span = range(0, deg2rad(20), length=5)
ϕ_span = range(0, 2π, length=5)
X, Y, Z = build_sphere_region(45.0, θ_span, ϕ_span)


it = dynamic_intensity(params, incident_field, t_span, X, Y, Z, 250_000)
save_matrix(intensity_file, it)
save_matrix(time_file, collect(t_span))
save_params(parameters_file, params, incident_field, 250_000)