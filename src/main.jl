import LinearAlgebra.BLAS: get_num_threads, set_num_threads
using BenchmarkTools
using ThreadPinning

set_num_threads(1)
if Sys.islinux()
    pinthreads(:numa)
    threadinfo(; color=false, slurm=true)
end

include("structs.jl")
include("utils.jl")
include("lattice.jl")
include("field.jl")
include("computation.jl")
include("averaging.jl")
include("misc.jl")

println("========== Configuration ==========")
println("Slurm JobId: $(get(ENV, "SLURM_JOB_ID", "local"))")
println(get_num_threads(), " threads used for BLAS operations.")
println(Threads.nthreads(), " threads used for Julia operations.")
println("Number of processes: ", nprocs())
println("=====================================")

# Simulation parameters
Na  = 1000
Nd  = 20
Rd  = 9.0
a   = 0.07
Δ0  = 0.0

# Electric field parameters
E0  = 1e-3
w0  = 4.0
θ   = deg2rad(10)
d   = bragg_periodicity(deg2rad(10))

incident_field = GaussianBeam(E0, w0, θ)
params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)

# plane_mean_intensity(params, incident_field, 2400, "data/2DStationnary-2/")
reflection_coefficient(params, incident_field, range(-1.0, 1.0, length=10), 100)