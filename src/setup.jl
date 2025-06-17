using InteractiveUtils
import Hwloc: num_physical_cores, num_virtual_cores
import LinearAlgebra.BLAS: get_num_threads

versioninfo()

println("========== Availaible configuration ==========")
println(num_physical_cores(), " physical cores detected.")
println(num_virtual_cores(), " logical cores detected.")
println(get_num_threads(), " threads available for BLAS operations.")
println(Threads.nthreads(), " threads available for Julia operations.")
println("=============================================")
