using CUDA, GPUArrays
using Polyester
using BenchmarkTools
using ProgressBars

# DO NOT execute if using the REPL, instantiate each function manually
include("structs.jl")
include("utils.jl")
include("lattice.jl")
include("field.jl")
include("computation.jl")

# =============== Averaging function ===============

"""
    mean_intensity(params::SimulationParameters{P, N}, incident_field::GaussianBeam{P}, X, Y, Z; iterations=100) where {P <: Real, N <: Integer}

Compute the mean scattered light intensity at specified observation points through Monte Carlo simulation.

This function performs multiple iterations of light scattering calculations to compute the ensemble-averaged
intensity distribution. Each iteration generates a new random configuration of scatterers in an optical
lattice and solves the multiple scattering problem.

# Arguments
- `params::SimulationParameters{P, N}`: Simulation parameters containing number of atoms (`Na`) and detuning (`Δ0`)
- `incident_field::GaussianBeam{P}`: The incident Gaussian beam illuminating the scatterers
- `X, Y, Z`: Coordinate arrays specifying the observation points where intensity is calculated
- `iterations=100`: Number of Monte Carlo iterations for ensemble averaging

# Returns
- `intensity`: Array of mean intensities at the observation points, normalized by the incident field intensity (`E0^2`)

# Algorithm
1. For each iteration:
   - Generate random scatterer positions in uniform optical lattice
   - Calculate inter-particle distances and system matrix
   - Compute incident field at scatterer locations
   - Solve linear system for scattering amplitudes
   - Calculate total field (incident + scattered) at observation points
   - Accumulate intensity contributions
2. Average over all iterations and normalize
"""
function mean_intensity(params::SimulationParameters, incident_field::GaussianBeam, X, Y, Z, iterations=100)
    scatterers = similar(X, params.Na, 3)
    M = similar(X, Complex{eltype(X)}, params.Na, params.Na)
    E = similar(X, Complex{eltype(X)}, params.Na)
    A = similar(X, Complex{eltype(X)}, params.Na)

    intensity = zero(X)
    scatt_intensity = similar(X, Complex{eltype(X)}, size(X, 1), size(X, 2))
    incident_intensity = similar(X, Complex{eltype(X)}, size(X, 1), size(X, 2))

    compute_field!(incident_intensity, X, Y, Z, incident_field)
    incident_intensity .*= convert(eltype(M), 0.5im)

    for i in ProgressBar(1:iterations)
        uniform_optical_lattice!(params, scatterers)
        compute_system_matrix!(M, scatterers, params.Δ0)

        compute_field!(E, view(scatterers, :, 1), view(scatterers, :, 2), view(scatterers, :, 3), incident_field)
        E .*= convert(eltype(M), 0.5im)

        A .= M \ E

        compute_scattered_field!(scatt_intensity, X, Y, Z, scatterers, A)
        intensity .+= abs2.(scatt_intensity + incident_intensity)

    end
    intensity ./= iterations .* incident_field.E0.^2  # Normalize by incident field intensity

    return intensity
end

function reflection_coeff(params::SimulationParameters, incident_field::GaussianBeam, iterations=100)
    θ_span = range(deg2rad(incident_field.θ - 1), deg2rad(incident_field.θ + 1), length=5)
    X, Y, Z = build_sphere_region(45.0, θ_span, [0.0, π])
end

function f(du, u, p, t)
    mul!(du, p, u)
    return nothing
end

function dynamic_intensity(params::SimulationParameters{P, N}, incident_field::GaussianBeam{P}, t_span, X, Y, Z; iterations=100) where {P <: Real, N <: Integer}
    scatterers = similar(X, params.Na, 3)
    M = similar(X, Complex{eltype(X)}, params.Na, params.Na)
    dist = similar(X, Float64, params.Na, params.Na)
    E = similar(X, Complex{eltype(X)}, params.Na)
    A = similar(X, Complex{eltype(X)}, params.Na)
    field = zero(X)
    intensity = zeros(length(t_span))

    for i in ProgressBar(1:iterations)
        uniform_optical_lattice!(params, scatterers)
        compute_system_matrix!(M, scatterers, params.Δ0)

        compute_field!(E, view(scatterers, :, 1), view(scatterers, :, 2), view(scatterers, :, 3), Ref(incident_field))

        if any(isinf.(real(M))) || any(isnan.(real(M)))
            @warn "System matrix contains infinite values, skipping iteration $i"
            display(M)
            continue
        end
        A .= M \ E

        problem = ODEProblem(f, A, (t_span[begin], t_span[end]), M)
        sol = solve(problem, Tsit5())

        k = 1
        for t in t_span
            A .= sol(t)
            field .= abs2.(compute_field.(X, Y, Z, Ref(incident_field)) .+ compute_scattered_field.(X, Y, Z, Ref(scatterers), Ref(A)))
            intensity[k] += sum(field) / length(field)
            k += 1
        end

    end
    intensity ./= intensity[1]

    return intensity
end

# ================== Main ==================
# Get the first command line argument
if length(ARGS) >= 1
    first_arg = ARGS[1]
    println("Slurm JOBID: $first_arg")
else
    println("No Slurm JOBID provided")
    first_arg = "none"
end

# Simulation parameters
Na  = 300
Nd  = 20
Rd  = 9.0
a   = 0.07
Δ0  = 0.0

# Electric field parameters
E0  = 1e-3
w0  = 4.0
θ   = deg2rad(10)
d   = bragg_periodicity(deg2rad(10))

# Others parameters
nb_iterations = 10000

incident_field = GaussianBeam(E0, w0, θ)
params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)

X, Z = build_xOz_plane(100, 90.0)

θ_span = range(deg2rad(incident_field.θ - 1), deg2rad(incident_field.θ + 1), length=5)
X, Y, Z = build_sphere_region(45.0, θ_span, [0.0, π])
# X_refl, Y_refl, Z_refl = build_sphere_region(45.0, θ_span, π)
X_d = cu(X)
Y_d = cu(Y)
Z_d = cu(Z)


i = 1
Δ0_span = range(-1.0, 2.0, length=51)
R = zeros(length(Δ0_span))
for i in axes(Δ0_span, 1)
    params = SimulationParameters(Na, Nd, Rd, a, d, Δ0_span[i])
    intensity = mean_intensity(params, incident_field, X_d, Y_d, Z_d, 3000)
    res = Array(sum(intensity, dims=1))
    R[i] = res[2] / res[1]  # Ratio of intensities at two points
end

# save_matrix_to_csv(R, "reflection_coefficients.csv")
# save_matrix_to_csv(collect(Δ0_span), "detuning_values.csv")

using Plots

plot(Δ0_span, R, title="Reflection Coefficient vs Detuning", xlabel="Detuning", ylabel="Reflection Coefficient", label="R", legend=:topright, color=:blue)
save_matrix_to_csv(R, "data/DetuningSearch-1/reflection_coefficients.csv")
save_matrix_to_csv(collect(Δ0_span), "data/DetuningSearch-1/detuning.csv")
save_params(params, incident_field, 3000, "data/DetuningSearch-1/parameters.txt")
# X_inc, Y_inc, Z_inc = build_sphere_region(45.0, range(deg2rad(169), deg2rad(171), length=5), range(0, 0, length=1))
# X_refl, Y_refl, Z_refl = build_sphere_region(45.0, range(deg2rad(169), deg2rad(171), length=5), range(π, π, length=1))

# heatmap(Z[:, 1], X[1, :], log10.(Array(intensity)'), title="Mean intensity distribution", xlabel="Z (m)", ylabel="X (m)", color=:jet, aspect_ratio=:equal)
# scatter!(vec(Z_inc), vec(X_inc), label="Incident point", color=:green, aspect_ratio=:equal)
# scatter!(vec(Z_refl), vec(X_refl), label="Reflection point", color=:blue, aspect_ratio=:equal)