using CUDA
using Polyester
using BenchmarkTools

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

# =============== System matrix computation functions ===============
function distance!(out, scatterers, factor=1.0)
    @inbounds @batch for i in axes(out, 2)
        xi = scatterers[i, 1]
        yi = scatterers[i, 2]
        zi = scatterers[i, 3]

        for j in 1:i
            diff = (xi - scatterers[j, 1])^2 +
                    (yi - scatterers[j, 2])^2 +
                    (zi - scatterers[j, 3])^2

            out[j, i] = factor * sqrt(diff)
        end
    end
end



function compute_system_matrix!(M, dist, Δ0)
    @batch for i in axes(M, 2)
        for j in 1:(i-1)
            @inbounds @fastmath M[j, i] = 0.5im * cispi(dist[j, i]) / (dist[j, i])
            M[i, j] = M[j, i]  # Symmetry
        end
    end

    @inbounds for i in axes(M, 1)
        M[i, i] = -0.5 + 1.0im * Δ0
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

Na = 1000
params = SimulationParameters{Float64, Int}(Na, 1, 1.0, 0.1, 0.2, 0.01)
scatterers = zeros(Na, 3)  # Example scatterers array
dist = zeros(Na, Na)  # Distance matrix
M = zeros(ComplexF64, Na, Na)  # System matrix

uniform_optical_lattice!(params, scatterers, true)
distance!(dist, scatterers, 1.0)
@benchmark compute_system_matrix!(M, dist, params.Δ0)
display(t)