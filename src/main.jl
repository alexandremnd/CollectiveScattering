using CUDA, GPUArrays
using Polyester
using BenchmarkTools
using ProgressBars
import LinearAlgebra.BLAS: get_num_threads

include("structs.jl")
include("utils.jl")
include("lattice.jl")
include("field.jl")
include("computation.jl")
include("averaging.jl")

const SLURM_JOB_ID::String = get(ENV, "SLURM_JOB_ID", "local")

println("========== Configuration ==========")
println("Slurm JOBID: $(SLURM_JOB_ID)")
println(BLAS.get_num_threads(), " threads used for BLAS operations.")
println(Threads.nthreads(), " threads used for Julia operations.")
pin_thread()
println("====================================")

exit()

# Simulation parameters
Na  = 300
Nd  = 20
Rd  = 9.0
a   = 0.07
Δ0  = 0.2

# Electric field parameters
E0  = 1e-3
w0  = 4.0
θ   = deg2rad(90)
d   = bragg_periodicity(deg2rad(10))

incident_field = GaussianBeam(E0, w0, θ)
params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)

t_span = 0:0.01:1.0  # Time span for dynamic intensity calculation
θ_span = range(0, 20, length=5)
ϕ_span = range(0, 2π, length=5)
X, Y, Z = build_sphere_region(45.0, θ_span, ϕ_span)
it = dynamic_intensity(params, incident_field, t_span, X, Y, Z, 1000000)

save_matrix_to_csv(it, "data/DynamicDetuned/intensity.csv")
save_params(params, incident_field, 1000000, "data/DynamicDetuned/parameters.txt")



# ===== Reflection Coefficient Search =====
# using Plots

# θ_span = range(deg2rad(incident_field.θ - 1), deg2rad(incident_field.θ + 1), length=5)
# X, Y, Z = build_sphere_region(45.0, θ_span, [0.0, π])
# # X_refl, Y_refl, Z_refl = build_sphere_region(45.0, θ_span, π)
# X_d = cu(X)
# Y_d = cu(Y)
# Z_d = cu(Z)


# i = 1
# Δ0_span = range(-1.0, 2.0, length=51)
# R = zeros(length(Δ0_span))
# for i in axes(Δ0_span, 1)
#     params = SimulationParameters(Na, Nd, Rd, a, d, Δ0_span[i])
#     intensity = mean_intensity(params, incident_field, X_d, Y_d, Z_d, 3000)
#     res = Array(sum(intensity, dims=1))
#     R[i] = res[2] / res[1]  # Ratio of intensities at two points
# end

# plot(Δ0_span, R, title="Reflection Coefficient vs Detuning", xlabel="Detuning", ylabel="Reflection Coefficient", label="R", legend=:topright, color=:blue)
# save_matrix_to_csv(R, "data/DetuningSearch-1/reflection_coefficients.csv")
# save_matrix_to_csv(collect(Δ0_span), "data/DetuningSearch-1/detuning.csv")
# save_params(params, incident_field, 3000, "data/DetuningSearch-1/parameters.txt")


# ===== Mean intensity + Angle check =====
# X_inc, Y_inc, Z_inc = build_sphere_region(45.0, range(deg2rad(169), deg2rad(171), length=5), range(0, 0, length=1))
# X_refl, Y_refl, Z_refl = build_sphere_region(45.0, range(deg2rad(169), deg2rad(171), length=5), range(π, π, length=1))

# heatmap(Z[:, 1], X[1, :], log10.(Array(intensity)'), title="Mean intensity distribution", xlabel="Z (m)", ylabel="X (m)", color=:jet, aspect_ratio=:equal)
# scatter!(vec(Z_inc), vec(X_inc), label="Incident point", color=:green, aspect_ratio=:equal)
# scatter!(vec(Z_refl), vec(X_refl), label="Reflection point", color=:blue, aspect_ratio=:equal)