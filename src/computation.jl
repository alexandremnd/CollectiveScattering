using GPUArrays
using Polyester

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