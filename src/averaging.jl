using DifferentialEquations
using LinearAlgebra

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
"""
function dynamic_intensity(params::SimulationParameters, incident_field::GaussianBeam, t_span, X, Y, Z, iterations=100)
    scatterers = similar(X, params.Na, 3)
    M = similar(X, Complex{eltype(X)}, params.Na, params.Na)
    E = similar(X, Complex{eltype(X)}, params.Na)
    A = similar(X, Complex{eltype(X)}, params.Na)

    intensity_t = zeros(size(X, 1), length(t_span))
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

        problem = ODEProblem(f, A, (t_span[begin], t_span[end]), M)
        sol = solve(problem, Tsit5())

        k = 1
        for t in t_span
            A .= sol(t)
            compute_scattered_field!(scatt_intensity, X, Y, Z, scatterers, A)
            intensity .= abs2.(scatt_intensity .+ incident_intensity)
            intensity_t[:, k] .+= sum(Array(intensity), dims=2) / size(intensity, 2)
            k += 1
        end

    end
    intensity_t ./= intensity_t[:, 1]

    return intensity_t
end