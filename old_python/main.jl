using BenchmarkTools
using Plots
using CUDA
using Distances
using LinearAlgebra
using ProgressBars

struct GaussianBeam{T<:Real}
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


function bragg_periodicity(θ)
    return 1.0 / (2.0 * cos(θ))
end

function optical_lattice!(Nd, Rd, a, d, scatterers; centered=true)
    @inbounds @fastmath @simd for i in axes(scatterers, 1)
        u = rand()
        theta = 2π * rand()
        disk_no = rand(0:(Nd-1))

        scatterers[i, 1] = Rd * sqrt(u) * cos(theta)
        scatterers[i, 2] = Rd * sqrt(u) * sin(theta)
        scatterers[i, 3] = disk_no * d + a * 2.0 * (rand() - 0.5) - a / 2.0

        if centered
            scatterers[i, 3] -= (Nd - 1) * d / 2.0
        end
    end

    return
end

function optical_lattice!(Nd, Rd, a, d, scatterers::CuDeviceMatrix{T, 1}; centered=true) where T <:Real
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i > size(scatterers, 1)
        return
    end

    u = rand(Float32)
    theta = 2.0f0π * rand(Float32)
    disk_no = rand(0:(Nd-1))

    @inbounds scatterers[i, 1] = Rd * sqrt(u) * cos(theta)
    @inbounds scatterers[i, 2] = Rd * sqrt(u) * sin(theta)
    @inbounds scatterers[i, 3] = disk_no * d + a * 2.0f0 * (rand(Float32) - 0.5f0) - a / 2.0f0

    if centered
        @inbounds scatterers[i, 3] -= (Nd - 1) * d / 2.0f0
    end

    return nothing
end

function compute_field(x, y, z, field::GaussianBeam{T})::ComplexF32 where T<:Real
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
    res = zero(Complex{eltype(x)}) # Ensure the precison matches the input precision

    for i in axes(scatterers, 1)
        @inbounds dist = (scatterers[i, 1] - x)^2 + (scatterers[i, 2] - y)^2 + (scatterers[i, 3] - z)^2
        dist = sqrt(dist)
        @inbounds res -= exp(2.0f0im * π * dist) / (2.0f0 * π * dist) * amplitudes[i]
    end

    return res
end

function compute_system_matrix!(scatterers, M, Δ0)
    pairwise!(Euclidean(), M, scatterers, dims=1)

    @. M = cispi(2. * M) / (2.0π * 1.0im * M)

    for i in 1:Na
        @inbounds M[i, i] = 1.0 - 2.0im * Δ0
    end

    M .*= -0.5
    M = Symmetric(M)

    return
end

function compute_system_matrix!(scatterers::CuDeviceMatrix{Float32, 1}, M::CuDeviceMatrix{ComplexF32, 1}, Δ0)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    m, n = size(M)
    if i > m || j > n
        return nothing
    end

    if i == j
        @inbounds M[i, j] = -0.5f0 + 1.0f0im * Δ0
        return nothing
    end

    dist = zero(eltype(M))
    for k in 1:3
        @inbounds diff = scatterers[i, k] - scatterers[j, k]
        dist += diff * diff
    end

    @inbounds M[i, j] = sqrt(dist)
    @inbounds M[i, j] = -0.5f0 * exp(2.0f0im * π * M[i, j]) / (2.0f0im * π * M[i, j])

    return
end


# ============ Main program ============
# Atomic parameters
Na  = 1000
Nd  = 50
Rd  = 9
a   = 0.01
Δ0  = 0f0


# Electric field parameters
E0  = 1e-3
w0  = 4.0
θ   = deg2rad(15)
d   = bragg_periodicity(deg2rad(15.0))
incident_field = GaussianBeam(E0, w0, θ)

# Final field parameters
Np  = 1000

# ============ Stationnary state computation ============
scatterers_d = CuMatrix{Float32}(undef, Na, 3)
M_d = CuMatrix{ComplexF32}(undef, Na, Na)
E_d = CuVector{ComplexF32}(undef, Na)
A_d = CuVector{ComplexF32}(undef, Na)
field_d = CUDA.zeros(Float32, Np, Np)# CuMatrix{Float32}(undef, Np, Np)

Z = range(-60f0, 60f0, length=Np)' .* ones(Np)
X = ones(Np)' .* range(-60f0, 60f0, length=Np)
X_d = CuMatrix{Float32}(X)
Z_d = CuMatrix{Float32}(Z)

for i in ProgressBar(1:1000)
    CUDA.@sync @cuda threads=1024 blocks=cld(Na, 1024) optical_lattice!(Nd, Rd, a, d, scatterers_d)
    @cuda threads=(32, 32) blocks=(cld(Na, 32), cld(Na, 32)) compute_system_matrix!(scatterers_d, M_d, Δ0)
    E_d .= 0.5f0im .* compute_field.(scatterers_d[:, 1], scatterers_d[:, 2], scatterers_d[:, 3], Ref(incident_field))
    CUDA.synchronize()

    A_d .= M_d \ E_d
    field_d .+= abs2.(compute_field.(X_d, 0f0, Z_d, Ref(incident_field)) .+ compute_scattered_field.(X_d, 0f0, Z_d, Ref(scatterers_d), Ref(A_d)))
end
# ============ Final field computation ============
field_d ./= 1000.0f0





# Create heatmap
heatmap(X[:,1], Z[1,:], Array(field_d),
    xlabel="Z", ylabel="X",
    title="Gaussian Beam Intensity",
    colorbar_scale=:log10,
    aspect_ratio=:equal,
    color=:thermal)

