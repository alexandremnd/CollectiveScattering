import LinearAlgebra.BLAS: get_num_threads, set_num_threads
using BenchmarkTools

const SLURM_JOB_ID::String = get(ENV, "SLURM_JOBID", "local")
set_num_threads(1)

include("structs.jl")
include("utils.jl")
include("lattice.jl")
include("field.jl")
include("computation.jl")
include("averaging.jl")
include("misc.jl")

println("========== Configuration ==========")
println("Slurm JOBID: $(SLURM_JOB_ID)")
println(get_num_threads(), " threads used for BLAS operations.")
println(Threads.nthreads(), " threads used for Julia operations.")
println("=====================================")


function plane_mean_intensity(params::SimulationParameters, incident_field::GaussianBeam, iterations=100, folder="data/default/")
    mkpath(folder)
    x_f = open(joinpath(folder, "X-$(SLURM_JOB_ID).txt"), "w")
    y_f = open(joinpath(folder, "Y-$(SLURM_JOB_ID).txt"), "w")
    z_f = open(joinpath(folder, "Z-$(SLURM_JOB_ID).txt"), "w")
    i_f = open(joinpath(folder, "intensity-$(SLURM_JOB_ID).txt"), "w")
    p_f = open(joinpath(folder, "parameters-$(SLURM_JOB_ID).txt"), "w")

    X, Y, Z = build_xOz_plane(200, 90.0)

    intensity = compute_mean_intensity(params, incident_field, X, Y, Z, iterations)

    save_params(p_f, params, incident_field, iterations)
    save_matrix(x_f, X)
    save_matrix(y_f, zero(X))
    save_matrix(z_f, Z)
    save_matrix(i_f, intensity)
end

function reflection_coefficient(params::SimulationParameters, incident_field::GaussianBeam, Δ0_span, iterations=100, folder="data/default/")
    mkpath(folder)
    Δ_f = open(joinpath(folder, "delta-$(SLURM_JOB_ID).txt"), "w")
    r_f = open(joinpath(folder, "intensity-$(SLURM_JOB_ID).txt"), "w")
    p_f = open(joinpath(folder, "parameters-$(SLURM_JOB_ID).txt"), "w")

    θ_span = range(incident_field.θ - deg2rad(1), incident_field.θ + deg2rad(1), length=5)
    ϕ_span = range(0, π, length=2)
    X, Y, Z = build_sphere_region(45.0, θ_span, ϕ_span)
    X_d, Y_d, Z_d = cu(X), cu(Y), cu(Z)

    R = zeros(length(Δ0_span))
    for i in axes(Δ0_span, 1)
        current_params = SimulationParameters(params.Na, params.Nd, params.Rd, params.a, params.d, Δ0_span[i])
        intensity = compute_mean_intensity(current_params, incident_field, X_d, Y_d, Z_d, iterations)

        res = sum(Array(intensity), dims=1)
        display(res)
        R[i] = res[2] / res[1]
    end

    save_params(p_f, params, incident_field, iterations)
    save_matrix(Δ_f, collect(Δ0_span))
    save_matrix(r_f, R)
end

# Simulation parameters
Na  = 1200
Nd  = 10
Rd  = 9.0
a   = 0.07
Δ0  = 0.0

# Electric field parameters
E0  = 1e-3
w0  = 4.0
θ   = deg2rad(10)
d   = bragg_periodicity(deg2rad(10))

incident_field = GaussianBeam(E0, w0, θ)
params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)


plane_mean_intensity(params, incident_field, 100, "data/2DStationnary-2/")
# reflection_coefficient(params, incident_field, range(-0.2, 0.3, length=10), 1000, "data/ReflectionCoefficient-3/")



# =========== Dynamic intensity computation ===========
# t_span = 0:0.05:1.0  # Time span for dynamic intensity calculation
# θ_span = range(0, deg2rad(20), length=5)
# ϕ_span = range(0, 2π, length=5)
# X, Y, Z = build_sphere_region(45.0, θ_span, ϕ_span)


# it = dynamic_intensity(params, incident_field, t_span, X, Y, Z, 250_000)
# save_matrix(intensity_file, it)
# save_matrix(time_file, collect(t_span))
# save_params(parameters_file, params, incident_field, 250_000)