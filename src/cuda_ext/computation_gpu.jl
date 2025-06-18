using CUDA, GPUArrays

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