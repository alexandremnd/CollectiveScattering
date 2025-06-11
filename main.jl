using CUDA, GPUArrays
using DifferentialEquations
using ProgressBars
using LinearAlgebra
using BenchmarkTools
using Plots


# =============== General utility functions ===============
struct SimulationParameters{T <: Real, N <: Integer}
    Na::N
    Nd::N
    Rd::T
    a::T
    d::T
    Δ0::T
end

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
function uniform_optical_lattice!(params::SimulationParameters{T, N}, scatterers, centered=true) where {T <: Real, N <: Integer}
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


function uniform_optical_lattice!(params::SimulationParameters{B, N}, scatterers::T, centered=true) where {B <: Real, N <: Integer, T <: AbstractGPUArray}
    CUDA.@sync @cuda threads=1024 blocks=cld(Na, 1024) _uniform_optical_lattice!(params, scatterers, centered)
end

function _uniform_optical_lattice!(params::SimulationParameters{B, N}, scatterers, centered=true) where {B <: Real, N <: Integer}
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

# =============== Beam and field computation functions ===============
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

function compute_field(x, y, z, field)
    proj_z = field.kx * x + field.ky * y + field.kz * z

    proj_r2 = (field.u2x * x + field.u2y * y + field.u2z * z)^2 +
              (field.u3x * x + field.u3y * y + field.u3z * z)^2

    zR = π * field.w0^2
    wZ = field.w0 * sqrt(1f0 + (proj_z / zR)^2)
    phase = atan(proj_z / zR)

    if proj_z == 0
        return field.E0 * exp(-proj_r2 / field.w0^2)
    end

    Rz = proj_z * (1f0 + (zR / proj_z)^2)
    E = field.E0 * (field.w0 / wZ) * exp(-proj_r2 / wZ^2) * exp(π * 2f0im * (proj_z + proj_r2 / (2 * Rz)) -1f0im * phase)
    return E
end

function compute_scattered_field(x, y, z, scatterers, amplitudes)
    res = zero(Complex{eltype(x)})

    @fastmath @inbounds for i in axes(scatterers, 1)
        dist = (scatterers[i, 1] - x)^2 + (scatterers[i, 2] - y)^2 + (scatterers[i, 3] - z)^2
        dist = sqrt(dist)
        res -= cispi(2.0 * dist) / (2.0 * π * dist) * amplitudes[i]
    end

    return res
end

# =============== System matrix computation functions ===============
function distance!(out, scatterers, factor=1.0)
    @inbounds @fastmath for i in axes(out, 2)
        xi = scatterers[i, 1]
        yi = scatterers[i, 2]
        zi = scatterers[i, 3]

        @simd for j in 1:i
            diff = (xi - scatterers[j, 1])^2 +
                    (yi - scatterers[j, 2])^2 +
                    (zi - scatterers[j, 3])^2

            out[j, i] = factor * sqrt(diff)
        end
    end
end

function compute_system_matrix!(M, dist, Δ0)
    @inbounds @fastmath for i in axes(M, 2)
        for j in 1:(i-1)
            M[j, i] = 0.5im * cis(dist[j, i]) / (dist[j, i])
            M[i, j] = M[j, i]  # Ensure symmetry
        end
    end

    @inbounds @fastmath @simd for i in axes(M, 1)
        M[i, i] = -0.5 + 1.0im * params.Δ0
    end
end

function compute_system_matrix!(scatterers::T, M, Δ0) where {T <: AbstractGPUArray}
    m = size(M, 1)
    @cuda threads=(32, 32) blocks=(cld(m, 32), cld(m, 32)) _compute_system_matrix!(scatterers, M, Δ0)
end

function _compute_system_matrix!(scatterers, M, Δ0)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y

    m, n = size(M)
    if i > m || j > n
        return nothing
    elseif i == j
        @inbounds M[i, j] = -0.5f0 + 1.0f0im * Δ0
    else
        dist = zero(eltype(M))

        k::Int32 = 1
        while k <= 3
            @inbounds diff = scatterers[i, k] - scatterers[j, k]
            dist += diff * diff

            k += one(k)
        end

        @fastmath dist = sqrt(dist)
        @inbounds @fastmath M[i, j] = -0.5f0 * exp(2.0f0im * π * dist) / (2.0f0im * π * dist)

    end
    return nothing
end

# =============== Averaging function ===============
"""
For a given simulation parameters, will average intensity of "iterations" configurations on CPU or GPU depending on memory location of X.
"""
function mean_intensity(params::SimulationParameters{P, N}, incident_field::GaussianBeam{P}, X, Y, Z; iterations=100) where {P <: Real, N <: Integer}
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

function f(du, u, p, t)
    mul!(du, p, u)
    return nothing
end

function dynamic_intensity(params::SimulationParameters{N, P}, incident_field::GaussianBeam{P}, t_span, X, Y, Z; iterations=100) where {P <: Real, N <: Integer}
    scatterers = similar(X, params.Na, 3)
    M = similar(X, Complex{eltype(X)}, params.Na, params.Na)
    dist = similar(X, Float64, params.Na, params.Na)
    E = similar(X, Complex{eltype(X)}, params.Na)
    A = similar(X, Complex{eltype(X)}, params.Na)
    field = zero(X)
    intensity = zeros(size(X, 1), length(t_span))

    for i in 1:iterations
        uniform_optical_lattice!(params, scatterers)
        compute_system_matrix!(M, dist, params.Δ0)

        E .= convert(eltype(M), 0.5im) .* compute_field.(view(scatterers, :, 1), view(scatterers, :, 2), view(scatterers, :, 3), Ref(incident_field))
        A .= M \ E

        problem = ODEProblem(f, A, (t_span[begin], t_span[end]), M)
        sol = solve(problem, Tsit5())

        k = 1
        for t in t_span
            A .= sol(t)
            field .= abs2.(compute_field.(X, Y, Z, Ref(incident_field)) .+ compute_scattered_field.(X, Y, Z, Ref(scatterers), Ref(A)))
            intensity[:, k] += sum(field, dims=2) / size(field, 2)
            k += 1
        end

    end
    intensity ./= intensity[:, 1]

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
θ   = deg2rad(30)
d   = bragg_periodicity(deg2rad(30))

# Others parameters
nb_iterations = 10000

# Computation

incident_field = GaussianBeam(E0, w0, θ)
params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)

if_d = GaussianBeam(Float32(E0), Float32(w0), Float32(θ))
params_d = SimulationParameters{Float32, Int32}(Na, Nd, Float32(Rd), Float32(a), Float32(d), Float32(Δ0))
X, Z = build_xOz_plane(350, 90)

X_d = CuMatrix{Float32}(X)
Z_d = CuMatrix{Float32}(Z)

intensity = mean_intensity(params, incident_field, X, 0.0, Z; iterations=nb_iterations)

using Plots

# Convert back to CPU for plotting
X_cpu = Array(X_d)
Z_cpu = Array(Z_d)
intensity_cpu = Array(intensity)

save_matrix_to_csv(intensity_cpu, "intensity_2D.csv")
save_matrix_to_csv(X_cpu, "X_2D.csv")
save_matrix_to_csv(Z_cpu, "Z_2D.csv")

heatmap(X_cpu[1, :], Z_cpu[:, 1], log10.(intensity_cpu'),
    xlabel="X", ylabel="Z", title="Intensity Distribution",
    aspect_ratio=:equal, color=:jet)