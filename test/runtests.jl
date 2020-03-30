import HankelTransforms
using Test

# Import GPU packages:
import CUDAapi

if CUDAapi.has_cuda()   # check that CUDA is installed
if CUDAapi.has_cuda_gpu()   # check that GPU is active
    try
        import CuArrays   # we have CUDA, so this should not fail
        CuArrays.allowscalar(false)   # disable slow fallback methods
    catch ex
        # something is wrong with the user's set-up (or there's a bug in CuArrays)
        @warn "CUDA is installed, but CuArrays.jl fails to load"
            exception = (ex, catch_backtrace())
    end
end
end

const gamma = 5.0


function mysinc(r)
    return sin(2 * pi * gamma * r) / (2 * pi * gamma * r)
end


function mysinc_spectrum(v, p)
    if (v >= 0) & (v < gamma)
        f2 = v^p * cos(p * pi / 2) /
             (2 * pi * gamma * sqrt(gamma^2 - v^2) *
             (gamma + sqrt(gamma^2 - v^2))^p)
    elseif v > gamma
        f2 = sin(p * asin(gamma / v)) / (2 * pi * gamma * sqrt(v^2 - gamma^2))
    end
    return f2
end


@testset "CPU" begin
    include("test_1d.jl")
end

@testset "GPU" begin
    if CUDAapi.has_cuda()   # check that CUDA is installed
    if CUDAapi.has_cuda_gpu()   # check that GPU is active
        include("test_1d_gpu.jl")
    end
    end
end
