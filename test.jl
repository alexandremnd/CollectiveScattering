using LinearAlgebra
using ProgressMeter
using BenchmarkTools

function test(C, A, B)
    Threads.@threads :static for i in 1:12
        @views C[:, i] .= A[:, :, i] \ B[:, i]
    end
end

println(Threads.nthreads(), " threads available")

N = 100

BLAS.set_num_threads(1)
res = @benchmark test($(zeros(N, 12)), $(rand(N, N, 12)), $(rand(N, 12)))
display(res)

BLAS.set_num_threads(12)
res = @benchmark $(rand(N, N)) \ $(rand(N))
display(res)