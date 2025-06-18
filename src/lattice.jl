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