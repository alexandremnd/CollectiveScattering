using ThreadPinning

function pin_thread()
    if Sys.islinux()
        IN_SLURM = get(ENV, "SLURM_JOBID", "local") != "local"
        ThreadPinning.pinthreads(:cores)
        ThreadPinning.openblas_pinthreads(:cores)
        ThreadPinning.threadinfo(; slurm=IN_SLURM, color=false)
        ThreadPinning.threadinfo(; blas=true, slurm=IN_SLURM, color=false)
    end
end

function save_matrix(file, matrix)
    for row in eachrow(matrix)
        println(file, join(row, ","))
    end
    close(file)
end

function save_params(file, params::SimulationParameters, incident_field::GaussianBeam, iterations)
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
    close(file)
end
