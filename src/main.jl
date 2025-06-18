import LinearAlgebra.BLAS: get_num_threads, set_num_threads
using BenchmarkTools
using Distributed
using ThreadPinning

const SLURM_JOB_ID::String = get(ENV, "SLURM_JOB_ID", "local")
const SLURM_NPROCS::Int = get(ENV, "SLURM_NPROCS", "1") |> parse(Int)

set_num_threads(1)
if Sys.islinux() && SLURM_JOB_ID != "local"
    addprocs(SLURM_NPROCS)
    distributed_pinthreads(:numa)
    distributed_getcpuids()
end

include("structs.jl")
include("utils.jl")
include("lattice.jl")
include("field.jl")
include("computation.jl")
include("averaging.jl")
include("misc.jl")

println("========== Configuration ==========")
println("Slurm JOBID: $(SLURM_JOB_ID)")
println(get_num_threads(), " threads used for BLAS operations.")
println(Threads.nthreads(), " threads used for Julia operations.")
println("Number of processes: ", nprocs())
println("=====================================")

# Simulation parameters
Na  = 300
Nd  = 10
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


plane_mean_intensity(params, incident_field, 1000, "data/2DStationnary-3/")
# reflection_coefficient(params, incident_field, range(-0.2, 0.3, length=10), 1000, "data/ReflectionCoefficient-3/")