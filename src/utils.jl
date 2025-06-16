

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

    return X, Z
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

function save_matrix_to_csv(matrix, filename)
    open(filename, "w") do file
        for row in eachrow(matrix)
            println(file, join(row, ","))
        end
    end
end

function save_params_to_csv(params::SimulationParameters, incident_field::GaussianBeam, iterations, filename)
    open(filename, "w") do file
        println(file, "Na: ", params.Na)
        println(file, "Nd: ", params.Nd)
        println(file, "Rd: ", params.Rd)
        println(file, "a: ", params.a)
        println(file, "d: ", params.d)
        println(file, "Δ0: ", params.Δ0)
        println(file, "E0: ", incident_field.E0)
        println(file, "w0: ", incident_field.w0)
        println(file, "θ: ", incident_field.θ)
        println(file, "Averaged iterations: ", iterations)
    end
end

# ========== Backend Computation Device Management ==========

abstract type AbstractBackend end

struct CPUBackend <: AbstractBackend end
struct GPUBackend <: AbstractBackend end

# Default backend
const DEFAULT_BACKEND = Ref{AbstractBackend}(CPUBackend())

get_backend() = DEFAULT_BACKEND[]
get_backend(x::AbstractArray) = CPUBackend()
get_backend(x::AbstractGPUArray) = GPUBackend()
set_backend!(backend::AbstractBackend) = (DEFAULT_BACKEND[] = backend)

# Array creation functions
Base.zeros(::Type{T}, backend::CPUBackend, dims...) where T = zeros(T, dims...)
Base.zeros(::Type{T}, backend::GPUBackend, dims...) where T = CUDA.zeros(T, dims...)

Base.ones(::Type{T}, backend::CPUBackend, dims...) where T = ones(T, dims...)
Base.ones(::Type{T}, backend::GPUBackend, dims...) where T = CUDA.ones(T, dims...)

