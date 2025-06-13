using CUDA, GPUArrays
using Polyester
using BenchmarkTools

# =============== Type definitions ===============
struct SimulationParameters{T <: Real, N <: Integer}
    Na::N
    Nd::N
    Rd::T
    a::T
    d::T
    Δ0::T
end

struct GaussianBeam{T <: Real}
    E0::T
    w0::T
    θ::T

    kx::T
    ky::T
    kz::T

    u2x::T
    u2y::T
    u2z::T

    u3x::T
    u3y::T
    u3z::T
end
GaussianBeam(E0, w0, θ) = GaussianBeam(E0, w0, θ, -sin(θ), zero(eltype(θ)), cos(θ), cos(θ), zero(eltype(θ)), sin(θ), zero(eltype(θ)), one(eltype(θ)), zero(eltype(θ)))

# =============== Utility functions ===============
function bragg_periodicity(θ)
    return convert(eltype(θ), 0.5) /  cos(θ)
end

function build_xOz_plane(Np, size)
    X = range(-size/2, size/2, length=Np)' .* ones(Np)
    Z = ones(Np)' .* range(-size/2, size/2, length=Np)

    return X, Z
end

function build_sphere_region(R, theta, phi)
    X = R .* sin.(theta) .* cos.(phi)'
    Y = R .* sin.(theta) .* sin.(phi)'
    Z = R .* cos.(theta) .* ones(length(phi))'

    return X, Y, Z
end

function save_matrix_to_csv(matrix, filename)
    open(filename, "w") do file
        for row in eachrow(matrix)
            println(file, join(row, ","))
        end
    end
end

# =============== Optical lattice generation functions ===============
"""
uniform_optical_lattice!(params::SimulationParameters, scatterers::AbstractMatrix, centered=true)

Generates a uniform optical lattice by randomly distributing scatterers across multiple disks.
CPU and GPU versions are provided (only CUDA available).

# Arguments
- `params::SimulationParameters`: Simulation parameters containing lattice dimensions and properties
- `scatterers::AbstractMatrix`: Output matrix to store scatterer positions (Na × 3)
- `centered::Bool=true`: Whether to center the lattice around z=0

# Description
This function generates `Na` scatterers distributed uniformly within disks of radius `Rd`.
Each scatterer is assigned to one of `Nd` disks spaced by distance `d`, with additional
random displacement within range `±a/2` along the z-axis.
"""
function uniform_optical_lattice!(params::SimulationParameters, scatterers::AbstractMatrix, centered=true)
    @inbounds @fastmath for i in axes(scatterers, 1)
        u = rand()
        theta = 2π * rand()
        disk_no = rand(0:(params.Nd -1))

        scatterers[i, 1] = params.Rd * sqrt(u) * cos(theta)
        scatterers[i, 2] = params.Rd * sqrt(u) * sin(theta)
        scatterers[i, 3] = disk_no * params.d + params.a * 2.0 * (rand() - 0.5) - params.a / 2.0

        if centered
            scatterers[i, 3] -= (params.Nd - 1) * params.d / 2.0
        end
    end

    return
end

function uniform_optical_lattice!(params::SimulationParameters, scatterers::AbstractGPUArray, centered=true)
    CUDA.@sync @cuda threads=1024 blocks=cld(Na, 1024) _uniform_optical_lattice!(params, scatterers, centered)
end

function _uniform_optical_lattice!(params::SimulationParameters, scatterers, centered=true)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if i > size(scatterers, 1)
        return nothing
    else
        u = rand(Float32)
        theta = 2.0f0π * rand(Float32)
        disk_no = rand(0:(params.Nd - 1))

        @inbounds @fastmath scatterers[i, Int32(1)] = params.Rd * sqrt(u) * cos(theta)
        @inbounds @fastmath scatterers[i, Int32(2)] = params.Rd * sqrt(u) * sin(theta)
        @inbounds scatterers[i, Int32(3)] = disk_no * params.d + params.a * 2.0f0 * (rand(Float32) - 0.5f0) - params.a / 2.0f0

        if centered
            @inbounds scatterers[i, Int32(3)] -= (params.Nd - one(params.Nd)) * params.d / 2.0f0
        end
    end

    return nothing
end

# =============== System matrix computation functions ===============
"""
Compute the system matrix for collective scattering calculations in-place.

This function fills the matrix `M` with the interaction coefficients between scatterers
in a collective scattering system. The matrix represents the coupling between different
scatterers based on their spatial positions and includes detuning effects.

# Arguments
- `M::AbstractMatrix`: Pre-allocated matrix to store the system matrix (modified in-place)
- `scatterers::AbstractMatrix`: Matrix where each row contains the (x, y, z) coordinates
  of a scatterer. Expected shape: (N_scatterers, 3)
- `Δ0::Real`: Detuning parameter that affects the diagonal elements

# Details
The function computes:
- Off-diagonal elements: `M[i,j] = 0.5im * exp(iπ * dist_ij) / dist_ij` where
  `dist_ij = 2 * ||r_i - r_j||` is the scaled distance between scatterers i and j
- Diagonal elements: `M[i,i] = -0.5 + 1.0im * Δ0`

The matrix is symmetric, so only the upper triangular part is computed and then
mirrored to the lower triangular part for efficiency.
"""
function compute_system_matrix!(M::AbstractMatrix, scatterers::AbstractMatrix, Δ0::Real)
    @batch for i in axes(M, 2)
        xi = scatterers[i, 1]
        yi = scatterers[i, 2]
        zi = scatterers[i, 3]
        for j in 1:(i-1)
            diff = (xi - scatterers[j, 1])^2 +
                    (yi - scatterers[j, 2])^2 +
                    (zi - scatterers[j, 3])^2

            dist = 2.0π * sqrt(diff)

            @inbounds @fastmath M[j, i] = 0.5im * cis(dist) / (dist)
            M[i, j] = M[j, i]  # Symmetry
        end
    end

    @inbounds for i in axes(M, 1)
        M[i, i] = -0.5 + 1.0im * Δ0
    end
end

function compute_system_matrix!(M::AbstractGPUArray, scatterers::AbstractGPUArray, Δ0::Real)
    m = size(M, 1)
    @cuda threads=(32, 32) blocks=(cld(m, 32), cld(m, 32)) _compute_system_matrix!(M, scatterers, Δ0)
end

function _compute_system_matrix!(M, scatterers, delta0)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    m, n = size(M)
    if i > m || j > n
        return nothing
    elseif i == j
        @inbounds M[i, j] = -0.5f0 + 1.0f0im * delta0
    else
        dist = zero(eltype(scatterers))

        diff = (scatterers[i, 1] - scatterers[j, 1])^2 +
               (scatterers[i, 2] - scatterers[j, 2])^2 +
               (scatterers[i, 3] - scatterers[j, 3])^2

        @fastmath dist = sqrt(diff)

        @fastmath @inbounds M[i, j] = -0.5f0 * exp(2.0f0im * π * dist) / (2.0f0im * π * dist)

    end
    return nothing
end

# =============== Field computation functions ===============

"""
    compute_field(x, y, z, field)

Compute the electric field amplitude of a Gaussian beam at a given position in 3D space.

This function calculates the complex electric field of a Gaussian beam propagating in an
arbitrary direction, taking into account beam divergence, wavefront curvature, and Gouy phase.

# Arguments
- `x::Real`: X-coordinate of the evaluation point
- `y::Real`: Y-coordinate of the evaluation point
- `z::Real`: Z-coordinate of the evaluation point
- `field`: Field object containing beam parameters with the following properties:
  - `kx, ky, kz`: Wave vector components defining propagation direction
  - `u2x, u2y, u2z`: Components of the second orthogonal unit vector (transverse direction)
  - `u3x, u3y, u3z`: Components of the third orthogonal unit vector (transverse direction)
  - `w0`: Beam waist radius at the focal point
  - `E0`: Electric field amplitude at the beam waist

# Returns
- `Complex`: Complex electric field amplitude at the specified position
"""
function compute_field(out, x, y, z, field::GaussianBeam{<:Real})
    @batch for i in axes(out, 2)
        for j in axes(out, 1)
            x = x[j, i]
            y = y[j, i]
            z = z[j, i]

            proj_z = field.kx * x + field.ky * y + field.kz * z

            proj_r2 = (field.u2x * x + field.u2y * y + field.u2z * z)^2 +
                    (field.u3x * x + field.u3y * y + field.u3z * z)^2

            zR = π * field.w0^2
            wZ = field.w0 * sqrt(1f0 + (proj_z / zR)^2)
            phase = atan(proj_z / zR)

            if proj_z == 0
                out[j, i] = field.E0 * exp(-proj_r2 / field.w0^2)
            end

            Rz = proj_z * (1f0 + (zR / proj_z)^2)
            E = field.E0 * (field.w0 / wZ) * exp(-proj_r2 / wZ^2) * exp(π * 2f0im * (proj_z + proj_r2 / (2 * Rz)) -1f0im * phase)

            out[j, i] = E
        end
    end
end

function compute_field(out::AbstractGPUArray, x::AbstractGPUArray, y::AbstractGPUArray, z::AbstractGPUArray, field::GaussianBeam{<:Real})
    m = length(x)
    @cuda threads=1024 blocks=cld(m, 1024) _compute_field!(vec(out), vec(x), vec(y), vec(z), field)
end

function _compute_field!(out, x, y, z, field)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if i > length(x)
        return nothing
    end

    x = x[i]
    y = y[i]
    z = z[i]

    proj_z = field.kx * x + field.ky * y + field.kz * z

    proj_r2 = (field.u2x * x + field.u2y * y + field.u2z * z)^2 +
              (field.u3x * x + field.u3y * y + field.u3z * z)^2

    if proj_z == 0
        @inbounds out[i] = field.E0 * exp(-proj_r2 / field.w0^2)
    else
        zR = π * field.w0^2
        @fastmath wZ = field.w0 * sqrt(1f0 + (proj_z / zR)^2)
        @fastmath phase = atan(proj_z / zR)

        @fastmath Rz = proj_z * (1f0 + (zR / proj_z)^2)
        E = field.E0 * (field.w0 / wZ) * exp(-proj_r2 / wZ^2 + π * 2f0im * (proj_z + proj_r2 / (2 * Rz)) - 1f0im * phase)

        @inbounds out[i] = E
    end
    return nothing
end


"""
    compute_scattered_field(x, y, z, scatterers, amplitudes)

Compute the total scattered electromagnetic field at a given point in 3D space.

# Arguments
- `x`, `y`, `z`: Coordinates of the observation point where the scattered field is evaluated
- `scatterers`: Matrix where each row contains the (x, y, z) coordinates of a scatterer
- `amplitudes`: Vector of complex scattering amplitudes for each scatterer

# Returns
- Complex-valued scattered field amplitude at the observation point

# Description
This function computes the sum of scattered waves from multiple point scatterers
using the free-space Green's function. Each scatterer contributes a spherical wave with
amplitude proportional to `amplitudes[i]` and phase determined by the distance from the
scatterer to the observation point.
"""
function compute_scattered_field(out, x, y, z, scatterers, amplitudes)
    @batch for i in axes(out, 2)
        for j in axes(out, 1)
            res = zero(eltype(out))

            for i in axes(scatterers, 1)
                dist = (scatterers[i, 1] - x)^2 + (scatterers[i, 2] - y)^2 + (scatterers[i, 3] - z)^2
                @fastmath dist = 2.0π * sqrt(dist)
                @fastmath res -= cis(dist) / (dist) * amplitudes[i]
            end

            @inbounds out[j, i] = res
        end
    end
end

function compute_scattered_field(out::AbstractGPUArray, x::AbstractGPUArray, y::AbstractGPUArray, z::AbstractGPUArray, scatterers::AbstractGPUArray, amplitudes::AbstractGPUArray)
    m = length(x)
    @cuda threads=1024 blocks=cld(m, 1024) _compute_scattered_field!(vec(out), vec(x), vec(y), vec(z), scatterers, amplitudes)
end

"""
Compute the scattered field at observation points using CUDA.
This is not a public function, but rather an internal implementation detail, do not call directly and use `compute_scattered_field` instead.
"""
function _compute_scattered_field!(out, x, y, z, scatterers, amplitudes)
    tx = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tx

    if i > length(out)
        return nothing
    else
        x_obs = x[i]
        y_obs = y[i]
        z_obs = z[i]

        field_sum = zero(eltype(out))
        num_scatterers = size(scatterers, 1)

        for j in 1:num_scatterers
            @inbounds dx = scatterers[j, 1] - x_obs
            @inbounds dy = scatterers[j, 2] - y_obs
            @inbounds dz = scatterers[j, 3] - z_obs

            dist_sq = dx*dx + dy*dy + dz*dz
            @fastmath dist = 2.0f0π * CUDA.sqrt(dist_sq)

            @fastmath @inbounds field_sum -= CUDA.cis(dist) / dist * amplitudes[j]
        end

        @inbounds out[i] = field_sum
    end
    return nothing
end

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
function mean_intensity(params::SimulationParameters, incident_field::GaussianBeam, X, Y, Z; iterations=100)
    scatterers = similar(X, params.Na, 3)
    M = similar(X, Complex{eltype(X)}, params.Na, params.Na)
    dist = similar(X, Float64, params.Na, params.Na)
    E = similar(X, Complex{eltype(X)}, params.Na)
    A = similar(X, Complex{eltype(X)}, params.Na)
    intensity = zero(X)

    for i in ProgressBar(1:iterations)
        uniform_optical_lattice!(params, scatterers)
        distance!(dist, scatterers, 2.0π)
        compute_system_matrix!(M, dist, params.Δ0)

        E .= convert(eltype(M), 0.5im) .* compute_field.(scatterers[:, 1], scatterers[:, 2], scatterers[:, 3], Ref(incident_field))

        A .= M \ E

        intensity .+= abs2.(compute_field.(X, Y, Z, Ref(incident_field)) .+ compute_scattered_field.(X, Y, Z, Ref(scatterers), Ref(A)))
    end
    intensity ./= iterations * incident_field.E0^2

    return intensity
end

# ================== Main ==================
# Simulation parameters
Na  = 1000
Nd  = 20
Rd  = 9.0
a   = 0.07
Δ0  = 0.0

# Electric field parameters
E0  = 1e-3
w0  = 4.0
θ   = deg2rad(90)
d   = bragg_periodicity(deg2rad(30))

# Others parameters
nb_iterations = 10000

incident_field = GaussianBeam(E0, w0, θ)
params = SimulationParameters(4, Nd, Rd, a, d, Δ0)

s_d = CuMatrix{Float32}(undef, Na, 3)
m_d = CuMatrix{ComplexF32}(undef, Na, Na)
e_d = CuVector{ComplexF32}(undef, Na)
a_d = CuVector{ComplexF32}(undef, Na)

uniform_optical_lattice!(params, s_d)
compute_system_matrix!(m_d, s_d, params.Δ0)
e_d .= convert(eltype(m_d), 0.5im) .* compute_field.(s_d[:, 1], s_d[:, 2], s_d[:, 3], Ref(incident_field))
a_d = m_d \ e_d


X, Z = build_xOz_plane(200, 90.0)
out = similar(X, ComplexF32)

x_d = cu(X)
z_d = cu(Z)
y_d = zero(x_d)
out_d = ComplexF32.(zero(x_d))
out_d1 = similar(out_d)
# @benchmark CUDA.@sync begin
#     compute_field(out_d, x_d, y_d, z_d, incident_field)
# end

shmem_bits = 512 * 3 * sizeof(Float32) + 512 * sizeof(ComplexF32)


@benchmark CUDA.@sync begin
    @cuda threads=512 blocks=cld(length(x_d), 512) shmem=shmem_bits t_gpu(vec(out_d1), vec(x_d), vec(y_d), vec(z_d), s_d, a_d, 512)
end


@benchmark CUDA.@sync begin
    @cuda threads=1024 blocks=cld(length(x_d), 1024) t_noshared(vec(out_d), vec(x_d), vec(y_d), vec(z_d), s_d, a_d)
end
    display(out_d)
display(out_d1)

