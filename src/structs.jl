"""
    SimulationParameters{T <: Real, N <: Integer}

A parametric struct for storing simulation parameters with type constraints.

# Type Parameters
- `T <: Real`: Numeric type for real-valued parameters (e.g., Float64, Float32)
- `N <: Integer`: Integer type for discrete parameters (e.g., Int64, Int32)

# Fields
- `Na`: Number of atoms
- `Nd`: Number of disks
- `Rd`: Radius of the disks
- `a`: Thickness of the disks
- `d`: Distance between disks
- `Δ0`: Detuning between the laser frequency and the atomic resonance frequency
"""
struct SimulationParameters{T <: Real, N <: Integer}
    Na::N
    Nd::N
    Rd::T
    a::T
    d::T
    Δ0::T
end


"""
    GaussianBeam{T <: Real}

A structure representing a Gaussian beam with real-valued parameters.

The Gaussian beam is a fundamental solution to the paraxial wave equation and is commonly
used in optics and laser physics to model the propagation of coherent light beams.

Do not instantiate this type directly; use the provided constructor `GaussianBeam(E0, w0, θ)`.
"""
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
GaussianBeam(E0, w0, θ) = GaussianBeam(E0, w0, θ, -sin(θ), zero(eltype(θ)), -cos(θ), cos(θ), zero(eltype(θ)), -sin(θ), zero(eltype(θ)), one(eltype(θ)), zero(eltype(θ)))