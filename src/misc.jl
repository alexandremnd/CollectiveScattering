"""
    bragg_periodicity(θ)

Calculate the Bragg periodicity for a given scattering angle.

# Arguments
- `θ`: Scattering angle (typically in radians)

# Returns
- Bragg periodicity value
"""
function bragg_periodicity(θ)
    return convert(eltype(θ), 0.5) /  cos(θ)
end


"""
    build_xOz_plane(Np, size)

Construct a plane in the xOz coordinate system.

# Arguments
- `Np`: Number of points to generate in the plane
- `size`: Size parameter defining the dimensions of the plane

# Returns
Returns a plane structure or array of points in the xOz plane.
First dimension is X varying, second dimension is Z varying.
"""
function build_xOz_plane(Np, size)
    X = range(-size/2, size/2, length=Np)' .* ones(Np)
    Z = ones(Np)' .* range(-size/2, size/2, length=Np)
    Y = zero(X)

    return X, Y, Z
end


"""
    build_sphere_region(R, theta, phi)

Build a spherical region with the specified parameters.

# Arguments
- `R`: Radius of the sphere
- `theta`: Polar angle (colatitude) in radians
- `phi`: Azimuthal angle in radians

# Returns
Returns a spherical region object or data structure representing the specified sphere.
First dimension is θ varying, second dimension is ϕ varying.
"""
function build_sphere_region(R, θ, ϕ)
    X = R .* sin.(θ) .* cos.(ϕ)'
    Y = R .* sin.(θ) .* sin.(ϕ)'
    Z = R .* cos.(θ) .* ones(length(ϕ))'

    return X, Y, Z
end