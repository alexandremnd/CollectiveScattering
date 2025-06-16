using LinearAlgebra
using Hwloc
using InteractiveUtils

versioninfo(verbose=true)

println(num_physical_cores(), " physical cores detected.")
println(num_virtual_cores(), " logical cores detected.")
println(BLAS.get_num_threads(), " threads available for BLAS operations.")
println(Threads.nthreads(), " threads available for Julia operations.")
