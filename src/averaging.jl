using OrdinaryDiffEq
import LinearAlgebra: mul!, diagind, LAPACK
using Distributed

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
function compute_mean_intensity(params::SimulationParameters, incident_field::GaussianBeam, X, Y, Z, iterations=100)
    scatterers = zeros(params.Na, 3)
    M = zeros(ComplexF64, params.Na, params.Na)
    E = zeros(ComplexF64, params.Na)
    A = zeros(ComplexF64, params.Na)

    intensity = zero(X)
    scatt_intensity = zeros(ComplexF64, size(X))
    incident_intensity = zeros(ComplexF64, size(X))

    compute_field!(incident_intensity, X, Y, Z, incident_field)
    incident_intensity .*= convert(eltype(M), 0.5im)

    for i in 1:iterations
        uniform_optical_lattice!(params, scatterers)
        compute_system_matrix!(M, scatterers, params.Δ0)

        compute_field!(E, view(scatterers, :, 1), view(scatterers, :, 2), view(scatterers, :, 3), incident_field)
        E .*= convert(eltype(M), 0.5im)

        A = M \ E

        compute_scattered_field!(scatt_intensity, X, Y, Z, scatterers, A)
        intensity .+= abs2.(scatt_intensity .+ incident_intensity)

    end
    intensity ./= iterations .* incident_field.E0.^2  # Normalize by incident field intensity

    return intensity
end

function f(du, u, p, t)
    mul!(du, p, u)
    return nothing
end


"""
    dynamic_intensity(params::SimulationParameters, incident_field::GaussianBeam, t_span, X, Y, Z; iterations=100)

Compute the dynamic intensity of scattered light over time for a given simulation setup.

# Arguments
- `params::SimulationParameters`: Simulation parameters containing physical constants and configuration
- `incident_field::GaussianBeam`: The incident Gaussian beam field
- `t_span`: Time span over which to calculate the dynamic intensity
- `X`: X-coordinate grid or position array
- `Y`: Y-coordinate grid or position array
- `Z`: Z-coordinate grid or position array
- `iterations=100`: Number of iterations for averaging (default: 100)

# Returns
The calculated dynamic intensity values over the specified time span and spatial coordinates.

# Description
This function computes the time-dependent intensity of light scattered by particles in the simulation
volume. The calculation is performed over the specified spatial grid (X, Y, Z) and time span,
with averaging performed over the specified number of iterations to improve statistical accuracy.

# Warning
Intensity is not normalized at all to easen parallelization, do no forget to normalize it afterwards if needed.
"""
function compute_dynamic_intensity(params::SimulationParameters, incident_field::GaussianBeam, t_span, X, Y, Z, iterations=100)
    scatterers = zeros(params.Na, 3)
    M = zeros(ComplexF64, params.Na, params.Na)
    E = zeros(ComplexF64, params.Na)
    A = zeros(ComplexF64, params.Na)

    intensity_t = zeros(size(X, 1), length(t_span))
    intensity = zero(X)
    scatt_intensity = zeros(ComplexF64, size(X))
    incident_intensity = zeros(ComplexF64, size(X))

    compute_field!(incident_intensity, X, Y, Z, incident_field)
    incident_intensity .*= convert(eltype(M), 0.5im)

    for i in 1:iterations
        uniform_optical_lattice!(params, scatterers)
        compute_system_matrix!(M, scatterers, params.Δ0)

        compute_field!(E, view(scatterers, :, 1), view(scatterers, :, 2), view(scatterers, :, 3), incident_field)
        E .*= convert(eltype(M), 0.5im)
        A .= M \ E

        M[diagind(M)] .-= 1.0im .* params.Δ0  # Remove detuning from diagonal (driving beam is off in dynamic case)

        problem = ODEProblem(f, A, (t_span[begin], t_span[end]), M)
        sol = solve(problem, Tsit5())

        k = 1
        for t in t_span
            A .= sol(t)
            compute_scattered_field!(scatt_intensity, X, Y, Z, scatterers, A)
            intensity .= abs2.(scatt_intensity .+ incident_intensity)
            intensity_t[:, k] .+= sum(intensity, dims=2) / size(intensity, 2)
            k += 1
        end

    end

    return intensity_t
end

"""
    plane_mean_intensity(params::SimulationParameters, incident_field::GaussianBeam, iterations=100, folder="data/default/")

Calculate the mean intensity distribution in a plane by averaging over multiple simulation iterations.

# Arguments
- `params::SimulationParameters`: Simulation configuration parameters
- `incident_field::GaussianBeam`: The incident Gaussian beam field
- `iterations::Integer=100`: Number of iterations to average over (default: 100)
- `folder::String="data/default/"`: Directory path for data storage (default: "data/default/"), directory will be created if it does not exist.

# Returns
- Mean intensity distribution in the specified plane

# Description
This function performs Monte Carlo averaging by running multiple scattering simulations
and computing the ensemble average of the intensity distribution. The averaging helps
reduce statistical noise and provides a more accurate representation of the mean
scattered field intensity. If Julia is launched with multiple threads, the averaging
is performed in parallel across the available threads (consider pinning threads to benefit from NUMA architecture).
"""
function plane_mean_intensity(params::SimulationParameters, incident_field::GaussianBeam, iterations=100, folder="data/default/")
    JOB_ID = get(ENV, "SLURM_JOB_ID", "local")
    mkpath(folder)
    x_f = open(joinpath(folder, filename("X")), "w")
    y_f = open(joinpath(folder, filename("Y")), "w")
    z_f = open(joinpath(folder, filename("Z")), "w")
    i_f = open(joinpath(folder, filename("intensity")), "w")
    p_f = open(joinpath(folder, filename("parameters")), "w")


    Threads.@threads :static for i in 1:Threads.nthreads()
        tmp_i = open(joinpath("/tmp/$(JOB_ID)", filename("intensity-$(i)")), "w")
        X, Y, Z = build_xOz_plane(200, 90.0)
        res = compute_mean_intensity(params, incident_field, X, Y, Z, cld(iterations, Threads.nthreads()))

        save_matrix(tmp_i, res)
    end

    # Combine results from all threads
    result_intensity = zeros(Float64, 200, 200)
    for i in 1:Threads.nthreads()
        res = load_matrix(joinpath("/tmp/$(JOB_ID)", filename("intensity-$(i)")))
        result_intensity .+= res ./ Threads.nthreads()
    end

    X, Y, Z = build_xOz_plane(200, 90.0)

    save_params(p_f, params, incident_field, iterations)
    save_matrix(x_f, X)
    save_matrix(y_f, zero(X))
    save_matrix(z_f, Z)
    save_matrix(i_f, result_intensity)
end


"""
    reflection_coefficient(params::SimulationParameters, incident_field::GaussianBeam, Δ0_span, iterations=100, folder="data/default/")

Calculate the reflection coefficient for collective scattering.

# Arguments
- `params::SimulationParameters`: Simulation parameters containing system configuration
- `incident_field::GaussianBeam`: The incident Gaussian beam field
- `Δ0_span`: Range or array of detuning values to analyze
- `iterations::Int=100`: Number of iterations for the calculation (default: 100)
- `folder::String="data/default/"`: Directory path for storing output data (default: "data/default/")

# Returns
The reflection coefficient as a function of the detuning parameters.

# Description
This function computes the reflection coefficient for different detuning (Δ0). If Julia is launched with multiple threads, the averaging
is performed in parallel across the available threads (consider pinning threads to benefit from NUMA architecture).
"""
function reflection_coefficient(params::SimulationParameters, incident_field::GaussianBeam, Δ0_span, iterations=100, folder="data/default/")
    mkpath(folder)
    Δ_f = open(joinpath(folder, filename("delta")), "w")
    r_f = open(joinpath(folder, filename("reflection")), "w")
    p_f = open(joinpath(folder, filename("parameters")), "w")

    R = zeros(length(Δ0_span))
    for i in axes(Δ0_span, 1)
        incident_intensity = Threads.Atomic{Float64}(0.0)
        reflected_intensity = Threads.Atomic{Float64}(0.0)

        # Certainly ensure the best scalability than threading Δ0_span since we don't always want to compute as much R(Δ0) as threads
        Threads.@threads :static for j in 1:Threads.nthreads()
            θ_span = range(incident_field.θ - deg2rad(1), incident_field.θ + deg2rad(1), length=5)
            ϕ_span = range(0, π, length=2)
            X, Y, Z = build_sphere_region(45.0, θ_span, ϕ_span)

            current_params = SimulationParameters(params.Na, params.Nd, params.Rd, params.a, params.d, Δ0_span[i])
            intensity = compute_mean_intensity(current_params, incident_field, X, Y, Z, cld(iterations, Threads.nthreads()))

            res = sum(intensity, dims=1)
            Threads.atomic_add!(incident_intensity, res[1])  # Assuming res[1] is the incident intensity and res[2] is the scattered intensity
            Threads.atomic_add!(reflected_intensity, res[2])
        end

        R[i] = reflected_intensity[] / incident_intensity[]
    end

    save_params(p_f, params, incident_field, iterations)
    save_matrix(Δ_f, collect(Δ0_span))
    save_matrix(r_f, R)
end


function dynamic_intensity(params::SimulationParameters, incident_field::GaussianBeam, t_span, radius, θ_span, ϕ_span, iterations=100, folder="data/default/")
    JOB_ID = get(ENV, "SLURM_JOB_ID", "local")
    mkpath(folder)
    t_f = open(joinpath(folder, filename("time")), "w")
    i_f = open(joinpath(folder, filename("intensity")), "w")
    p_f = open(joinpath(folder, filename("parameters")), "w")
    θ_f = open(joinpath(folder, filename("theta")), "w")

    Threads.@threads for i in 1:Threads.nthreads()
        tmp_i = open(joinpath("/tmp/$(JOB_ID)", filename("intensity-$(i)")), "w")
        X, Y, Z = build_sphere_region(radius, θ_span, ϕ_span)
        res = compute_dynamic_intensity(params, incident_field, t_span, X, Y, Z, cld(iterations, Threads.nthreads()))

        save_matrix(tmp_i, res)
    end

    # Combine results from all threads (we could use a more efficient way to do this such as using a thread lock,
    # but as this is not the longest operation, we can afford to do it this way)
    intensity_t = zeros(Float64, length(θ_span), length(t_span))
    for i in 1:Threads.nthreads()
        res = load_matrix(joinpath("/tmp/$(JOB_ID)", filename("intensity-$(i)")))
        intensity_t .+= res ./ Threads.nthreads()
    end

    intensity_t ./= intensity_t[:, 1]

    save_params(p_f, params, incident_field, iterations)
    save_matrix(t_f, collect(t_span))
    save_matrix(i_f, intensity_t)
    save_matrix(θ_f, collect(θ_span))
end