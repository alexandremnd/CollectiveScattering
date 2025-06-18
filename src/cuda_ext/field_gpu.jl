using CUDA, GPUArrays

function compute_field!(out::AbstractGPUArray, x::AbstractGPUArray, y::AbstractGPUArray, z::AbstractGPUArray, field::GaussianBeam{<:Real})
    m = length(x)
    @cuda threads=1024 blocks=cld(m, 1024) _compute_field!(vec(out), vec(x), vec(y), vec(z), field)
end

function _compute_field!(out, x, y, z, field)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if i > length(x)
        return nothing
    end

    x = x[i]
    y = y[i]
    z = z[i]

    proj_z = field.kx * x + field.ky * y + field.kz * z

    proj_r2 = (field.u2x * x + field.u2y * y + field.u2z * z)^2 +
              (field.u3x * x + field.u3y * y + field.u3z * z)^2

    if proj_z == 0
        @inbounds out[i] = field.E0 * exp(-proj_r2 / field.w0^2)
    else
        zR = π * field.w0^2
        @fastmath wZ = field.w0 * sqrt(1f0 + (proj_z / zR)^2)
        @fastmath phase = atan(proj_z / zR)

        @fastmath Rz = proj_z * (1f0 + (zR / proj_z)^2)
        E = field.E0 * (field.w0 / wZ) * exp(-proj_r2 / wZ^2 + π * 2f0im * (proj_z + proj_r2 / (2 * Rz)) - 1f0im * phase)

        @inbounds out[i] = E
    end
    return nothing
end


function compute_scattered_field!(out::AbstractGPUArray, x::AbstractGPUArray, y::AbstractGPUArray, z::AbstractGPUArray, scatterers::AbstractGPUArray, amplitudes::AbstractGPUArray)
    m = length(x)
    @cuda threads=1024 blocks=cld(m, 1024) _compute_scattered_field!(vec(out), vec(x), vec(y), vec(z), scatterers, amplitudes)
end


# ============ Scattered Field Computation ============


function _compute_scattered_field!(out, x, y, z, scatterers, amplitudes)
    tx = threadIdx().x
    i = (blockIdx().x - 1) * blockDim().x + tx

    if i > length(out)
        return nothing
    else
        x_obs = x[i]
        y_obs = y[i]
        z_obs = z[i]

        field_sum = zero(eltype(out))
        num_scatterers = size(scatterers, 1)

        for j in 1:num_scatterers
            @inbounds dx = scatterers[j, 1] - x_obs
            @inbounds dy = scatterers[j, 2] - y_obs
            @inbounds dz = scatterers[j, 3] - z_obs

            dist_sq = dx*dx + dy*dy + dz*dz
            @fastmath dist = 2.0f0π * CUDA.sqrt(dist_sq)

            @fastmath @inbounds field_sum -= CUDA.cis(dist) / dist * amplitudes[j]
        end

        @inbounds out[i] = field_sum
    end
    return nothing
end