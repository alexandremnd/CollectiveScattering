abstract type AbstractBackend end

struct CPUBackend <: AbstractBackend end
struct GPUBackend <: AbstractBackend end

# Default backend
const DEFAULT_BACKEND = Ref{AbstractBackend}(CPUBackend())

get_backend() = DEFAULT_BACKEND[]
get_backend(x::AbstractArray) = CPUBackend()
get_backend(x::AbstractGPUArray) = GPUBackend()
set_backend!(backend::AbstractBackend) = (DEFAULT_BACKEND[] = backend)

# Array creation functions
Base.zeros(::Type{T}, backend::CPUBackend, dims...) where T = zeros(T, dims...)
Base.zeros(::Type{T}, backend::GPUBackend, dims...) where T = CUDA.zeros(T, dims...)

Base.ones(::Type{T}, backend::CPUBackend, dims...) where T = ones(T, dims...)
Base.ones(::Type{T}, backend::GPUBackend, dims...) where T = CUDA.ones(T, dims...)