using Distributed
import LinearAlgebra.BLAS: get_num_threads, set_num_threads
using ThreadPinning

# Add worker processes if not already done
if nprocs() == 1
    addprocs(28)  # Adjust number based on your CPU cores
end

# Load required modules on all workers
@everywhere begin
    using CUDA, GPUArrays
    using Polyester
    import LinearAlgebra.BLAS: get_num_threads, set_num_threads
    using ThreadPinning

    include("structs.jl")
    include("utils.jl")
    include("lattice.jl")
    include("field.jl")
    include("computation.jl")
    include("averaging.jl")
    include("misc.jl")
end

@everywhere function run_simulation(run_id::Int)
    SLURM_JOB_ID::String = get(ENV, "SLURM_JOBID", "local")

    # Create unique file names for each run
    intensity_file = open("data/test/intensity-$(SLURM_JOB_ID)-run$(run_id).csv", "w")
    parameters_file = open("data/test/parameters-$(SLURM_JOB_ID)-run$(run_id).txt", "w")
    time_file = open("data/test/time-$(SLURM_JOB_ID)-run$(run_id).txt", "w")

    try
        set_num_threads(1)

        # Simulation parameters
        Na  = 1000
        Nd  = 20
        Rd  = 9.0
        a   = 0.07
        Δ0  = 0.2

        # Electric field parameters
        E0  = 1e-3
        w0  = 4.0
        θ   = deg2rad(90)
        d   = bragg_periodicity(deg2rad(10))

        incident_field = GaussianBeam(E0, w0, θ)
        params = SimulationParameters(Na, Nd, Rd, a, d, Δ0)

        t_span = 0:0.05:1.0
        θ_span = range(0, deg2rad(20), length=5)
        ϕ_span = range(0, 2π, length=5)
        X, Y, Z = build_sphere_region(45.0, θ_span, ϕ_span)

        # Run simulation
        it = compute_dynamic_intensity(params, incident_field, t_span, X, Y, Z, 1000)

        # Save results
        save_matrix(intensity_file, it)
        save_matrix(time_file, collect(t_span))
        save_params(parameters_file, params, incident_field, 10000)

        println("Completed simulation run $run_id")
        return run_id

    finally
        close(intensity_file)
        close(parameters_file)
        close(time_file)
    end
end

# Main execution
function main()

    println("========== Parallel Configuration ==========")
    println("Number of processes: $(nprocs())")
    println("Number of workers: $(nworkers())")
    println("==========================================")

    N = 1  # Number of parallel runs - adjust as needed
    ThreadPinning.distributed_pinthreads(:numa)

    # Run N simulations in parallel
    println("Starting $N parallel simulations...")
    results = pmap(run_simulation, 1:N)

    println("All simulations completed!")
    println("Completed runs: $results")
end

# Run the parallel simulations
main()