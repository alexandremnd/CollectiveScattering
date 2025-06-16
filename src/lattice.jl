"""
uniform_optical_lattice!(params::SimulationParameters, scatterers::AbstractMatrix, centered=true)

Generates a uniform optical lattice by randomly distributing scatterers across multiple disks.
CPU and GPU versions are provided (only CUDA available).

# Arguments
- `params::SimulationParameters`: Simulation parameters containing lattice dimensions and properties
- `scatterers::AbstractMatrix`: Output matrix to store scatterer positions (Na × 3)
- `centered::Bool=true`: Whether to center the lattice around z=0

# Description
This function generates `Na` scatterers distributed uniformly within disks of radius `Rd`.
Each scatterer is assigned to one of `Nd` disks spaced by distance `d`, with additional
random displacement within range `±a/2` along the z-axis.
"""
function uniform_optical_lattice!(params::SimulationParameters, scatterers::AbstractMatrix, centered=true)
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

function uniform_optical_lattice!(params::SimulationParameters, scatterers::AbstractGPUArray, centered=true)
    CUDA.@sync @cuda threads=1024 blocks=cld(Na, 1024) _uniform_optical_lattice!(params, scatterers, centered)
end

function _uniform_optical_lattice!(params::SimulationParameters, scatterers, centered=true)
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