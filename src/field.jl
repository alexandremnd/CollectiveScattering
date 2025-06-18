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
function compute_field!(out, x, y, z, field::GaussianBeam{<:Real})
    out .= _compute_field.(x, y, z, Ref(field))
end

function _compute_field(x, y, z, field::GaussianBeam{<:Real})
    proj_z = field.kx * x + field.ky * y + field.kz * z

    proj_r2 = (field.u2x * x + field.u2y * y + field.u2z * z)^2 +
            (field.u3x * x + field.u3y * y + field.u3z * z)^2

    zR = π * field.w0^2
    wZ = field.w0 * sqrt(1.0 + (proj_z / zR)^2)
    phase = atan(proj_z / zR)

    if proj_z == 0
        return field.E0 * exp(-proj_r2 / field.w0^2)
    end

    Rz = proj_z * (1 + (zR / proj_z)^2)
    E = field.E0 * (field.w0 / wZ) * exp(-proj_r2 / wZ^2) * exp(π * 2.0im * (proj_z + proj_r2 / (2.0 * Rz)) - 1.0im * phase)

    return E
end

# ============ Scattered Field Computation ============

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
function compute_scattered_field!(out, x, y, z, scatterers, amplitudes)
    for i in axes(out, 2)
        for j in axes(out, 1)
            res = zero(eltype(out))

            for k in axes(scatterers, 1)
                dist = (scatterers[k, 1] - x[j, i])^2 + (scatterers[k, 2] - y[j, i])^2 + (scatterers[k, 3] - z[j, i])^2
                @fastmath dist = 2.0π * sqrt(dist)
                @fastmath res -= cis(dist) / (dist) * amplitudes[k]
            end
            @inbounds out[j, i] = res
        end
    end
end