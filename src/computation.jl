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
    for i in axes(M, 2)
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