import HankelTransforms

import CUDAapi
using Test

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
    include("test_2d.jl")
end

@testset "GPU" begin
    # check that CUDA is installed and GPU is active:
    if CUDAapi.has_cuda_gpu()
        import CuArrays
        CuArrays.allowscalar(false)   # disable slow fallback methods
        include("test_1d_gpu.jl")
        include("test_2d_gpu.jl")
    end
end
