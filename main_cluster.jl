using Polyester
using BenchmarkTools

struct SimulationParameters{T <: Real, N <: Integer}
    Na::N
    Nd::N
    Rd::T
    a::T
    d::T
    Δ0::T
end


function uniform_optical_lattice!(params::SimulationParameters{T, N}, scatterers, centered=true) where {T <: Real, N <: Integer}
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

function distance!(out, scatterers, factor=1.0)
    @inbounds @batch for i in axes(out, 2)
        xi = scatterers[i, 1]
        yi = scatterers[i, 2]
        zi = scatterers[i, 3]

        for j in 1:i
            diff = (xi - scatterers[j, 1])^2 +
                    (yi - scatterers[j, 2])^2 +
                    (zi - scatterers[j, 3])^2

            out[j, i] = factor * sqrt(diff)
        end
    end
end



function compute_system_matrix!(M, dist, Δ0)
    @batch for i in axes(M, 2)
        for j in 1:(i-1)
            @inbounds @fastmath M[j, i] = 0.5im * cis(dist[j, i]) / (dist[j, i])
        end
    end

    @inbounds for i in axes(M, 1)
        M[i, i] = -0.5 + 1.0im * Δ0
    end
end



Na = 1000
params = SimulationParameters{Float64, Int}(Na, 1, 1.0, 0.1, 0.2, 0.01)
scatterers = zeros(Na, 3)  # Example scatterers array
dist = zeros(Na, Na)  # Distance matrix
M = zeros(ComplexF64, Na, Na)  # System matrix

uniform_optical_lattice!(params, scatterers, true)
distance!(dist, scatterers, 1.0)
@benchmark compute_system_matrix!(M, dist, params.Δ0)
display(t)
