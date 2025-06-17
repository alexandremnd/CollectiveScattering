using Hwloc
using InteractiveUtils

versioninfo(verbose=true)

println("========== Availaible configuration ==========")
println(num_physical_cores(), " physical cores detected.")
println(num_virtual_cores(), " logical cores detected.")
println(LinearAlgebra.BLAS.get_num_threads(), " threads available for BLAS operations.")
println(Threads.nthreads(), " threads available for Julia operations.")
println("=============================================")

if !Sys.islinux()
    atexit(() -> run("export JULIA_NUM_THREADS=$(BLAS.get_num_threads())"))
end