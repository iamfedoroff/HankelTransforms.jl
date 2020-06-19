module HankelTransforms

import Base: *, \
import JLD2
import LinearAlgebra


# CPU / GPU specific functions -------------------------------------------------
import CUDA

isongpu = function(T)
    return false
end

cuconvert = function(F)
    return F
end

# check that CUDA is installed and GPU is active:
if CUDA.has_cuda_gpu()
    CUDA.allowscalar(false)   # disable slow fallback methods

    global isongpu = function(T)
        if T <: CUDA.CuArray
            IOG = true
        else
            IOG = false
        end
        return IOG
    end

    global cuconvert = function(F)
        return CUDA.CuArray(F)
    end
end


# Bessel functions -------------------------------------------------------------
# GSL is the Julia wrapper for the GNU Scientific Library (GSL) and does not
# work under Windows.
# import GSL
# bessel(p, x) = GSL.sf_bessel_Jn(p, x)
# bessel_zero(p, n) = GSL.sf_bessel_zero_Jnu(p, n)

import FunctionZeros
import SpecialFunctions
bessel(p, x) = SpecialFunctions.besselj(p, x)
bessel_zero(p, n) = FunctionZeros.besselj_zero(p, n)
# ------------------------------------------------------------------------------


export plan, htcoord, htfreq, dht!, dht, idht!, idht

const htAbstractArray{T} = Union{AbstractArray{T}, AbstractArray{Complex{T}}}


struct Plan{
    IOG,
    I<:Int,
    CI<:CartesianIndices,
    T<:AbstractFloat,
    UJ<:AbstractArray{T},
    UT<:AbstractArray{T},
    UF<:htAbstractArray{T},
}
    N :: I
    region :: CI
    R :: T
    V :: T
    J :: UJ
    TT :: UT
    ftmp :: UF
end


function plan(
    R::T, F::UF, p::Int=0; kwargs...,
) where {T<:AbstractFloat, UF<:htAbstractArray{T}}
    region = CartesianIndices(F)
    return plan(R, F, region, p; kwargs...)
end


function plan(
    R::T,
    F::UF,
    region::CartesianIndices,
    p::Int=0;
    save::Bool=false,
    fname::String="dht.jld2",
) where {T<:AbstractFloat, UF<:htAbstractArray{T}}
    dims = size(region)
    N = dims[1]

    a = zeros(T, N)
    J = zeros(T, N)
    TT = zeros(T, (N, N))

    @. a = bessel_zero(p, 1:N)
    aNp1::T = bessel_zero(p, N + 1)

    V::T = aNp1 / (2 * pi * R)
    @. J = abs(bessel(p + 1, a)) / R

    S::T = 2 * pi * R * V

    for j=1:N
    for i=1:N
        TT[i, j] = 2 * bessel(p, a[i] * a[j] / S) /
                   abs(bessel(p + 1, a[i])) /
                   abs(bessel(p + 1, a[j])) / S
    end
    end

    if save
        fp = JLD2.jldopen(fname, "w")
        fp["T"] = T
        fp["UF"] = UF
        fp["N"] = N
        fp["region"] = region
        fp["R"] = R
        fp["V"] = V
        fp["J"] = J
        fp["TT"] = TT
        JLD2.close(fp)
    end

    ftmp = zeros(T, dims)

    IOG = isongpu(UF)
    if IOG
        J = cuconvert(J)
        TT = cuconvert(TT)
        ftmp = cuconvert(ftmp)
    end

    I = typeof(N)
    CI = typeof(region)
    UJ = typeof(J)
    UT = typeof(TT)
    return Plan{IOG, I, CI, T, UJ, UT, UF}(N, region, R, V, J, TT, ftmp)
end


function plan(fname::String)
    fp = JLD2.jldopen(fname, "r")
    T = fp["T"]
    UF = fp["UF"]
    N = fp["N"]
    region = fp["region"]
    R = fp["R"]
    V = fp["V"]
    J = fp["J"]
    TT = fp["TT"]
    JLD2.close(fp)

    dims = size(region)
    ftmp = zeros(T, dims)

    IOG = isongpu(UF)
    if IOG
        J = cuconvert(J)
        TT = cuconvert(TT)
        ftmp = cuconvert(ftmp)
    end

    I = typeof(N)
    CI = typeof(region)
    UJ = typeof(J)
    UT = typeof(TT)
    return Plan{IOG, I, CI, T, UJ, UT, UF}(N, region, R, V, J, TT, ftmp)
end


"""
Compute the spatial coordinates for Hankel transform.
"""
function htcoord(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = zeros(T, N)
    @. a = bessel_zero(p, 1:N)
    aNp1::T = bessel_zero(p, N + 1)
    V::T = aNp1 / (2 * pi * R)
    @. a = a / (2 * pi * V)   # resuse the same array to avoid allocations
    return a
end


"""
Compute the spatial frequencies (ordinary, not angular) for Hankel transform.
"""
function htfreq(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = zeros(T, N)
    @. a = bessel_zero(p, 1:N)
    @. a = a / (2 * pi * R)   # resuse the same array to avoid allocations
    return a
end


# ******************************************************************************
# AbstractFFTs API
# ******************************************************************************
function *(plan::Plan, f::AbstractArray)
    dht!(f, plan)
    return nothing
end


function \(plan::Plan, f::AbstractArray)
    idht!(f, plan)
    return nothing
end


# ******************************************************************************
# Out of place functions
# ******************************************************************************
"""
Compute (out of place) forward discrete Hankel transform.
"""
function dht(
    f::UF, plan::Plan{IOG, I, CI, T, UJ, UT, UF},
) where {IOG, I, CI, T, UJ, UT, UF}
    ftmp = copy(f)
    dht!(ftmp, plan)
    return ftmp
end


"""
Compute (out of place) backward discrete Hankel transform.
"""
function idht(
    f::UF, plan::Plan{IOG, I, CI, T, UJ, UT, UF},
) where {IOG, I, CI, T, UJ, UT, UF}
    ftmp = copy(f)
    idht!(ftmp, plan)
    return ftmp
end


# ******************************************************************************
# CPU functions
# ******************************************************************************
"""
Compute (in place) forward discrete Hankel transform on CPU.
"""
function dht!(
    f::UF, plan::Plan{false, I, CI, T, UJ, UT, UF},
) where {I, CI, T, UJ, UT, UF}
    kernel1(f, plan.J, plan.R, plan.region)
    kernel2(f, plan.ftmp, plan.TT, plan.region)
    kernel3(f, plan.ftmp, plan.J, plan.V, plan.region)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on CPU.
"""
function idht!(
    f::UF, plan::Plan{false, I, CI, T, UJ, UT, UF},
) where {I, CI, T, UJ, UT, UF}
    kernel1(f, plan.J, plan.V, plan.region)
    kernel2(f, plan.ftmp, plan.TT, plan.region)
    kernel3(f, plan.ftmp, plan.J, plan.R, plan.region)
    return nothing
end


function kernel1(f, J, RV, region)
    # axis = 1
    N = length(region)
    for k=1:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds f[k] = f[k] * RV / J[i]
    end
    return nothing
end


function kernel2(f::AbstractArray{T, 1}, ftmp, TT, region) where T
    # axis = 1
    N = length(region)
    Naxis = size(region)[1]   # Naxis = size(region)[axis]
    for k=1:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m]
        end
    end
    return nothing
end


function kernel2(f::AbstractArray{T, 2}, ftmp, TT, region) where T
    # axis = 1
    N = length(region)
    Naxis = size(region)[1]   # Naxis = size(region)[axis]
    for k=1:N
        i = region[k][1]   # i = region[k][axis]
        j = region[k][2]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m, j]
        end
    end
    return nothing
end


function kernel3(f, ftmp, J, RV, region)
    # axis = 1
    N = length(region)
    for k=1:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds f[k] = ftmp[k] * J[i] / RV
    end
    return nothing
end


# ******************************************************************************
# GPU functions
# ******************************************************************************
"""
Compute (in place) forward discrete Hankel transform on GPU.
"""
function dht!(
    f::UF, plan::Plan{true, I, CI, T, UJ, UT, UF},
) where {I, CI, T, UJ, UT, UF}
    N = length(plan.region)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDA.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDA.@cuda config=get_config kernel1(f, plan.J, plan.R, plan.region)
    CUDA.@cuda config=get_config kernel2(f, plan.ftmp, plan.TT, plan.region)
    CUDA.@cuda config=get_config kernel3(f, plan.ftmp, plan.J, plan.V, plan.region)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on GPU.
"""
function idht!(
    f::UF, plan::Plan{true, I, CI, T, UJ, UT, UF},
) where {I, CI, T, UJ, UT, UF}
    N = length(plan.region)

    function get_config(kernel)
        fun = kernel.fun
        config = CUDA.launch_configuration(fun)
        blocks = cld(N, config.threads)
        return (threads=config.threads, blocks=blocks)
    end

    CUDA.@cuda config=get_config kernel1(f, plan.J, plan.V, plan.region)
    CUDA.@cuda config=get_config kernel2(f, plan.ftmp, plan.TT, plan.region)
    CUDA.@cuda config=get_config kernel3(f, plan.ftmp, plan.J, plan.R, plan.region)
    return nothing
end


function kernel1(f::CUDA.CuDeviceArray, J, RV, region)
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    # axis = 1
    N = length(region)
    for k=id:stride:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds f[k] = f[k] * RV / J[i]
    end
    return nothing
end


function kernel2(f::CUDA.CuDeviceArray{T, 1}, ftmp, TT, region) where T
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    # axis = 1
    N = length(region)
    Naxis = size(region)[1]   # Naxis = size(region)[axis]
    for k=id:stride:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m]
        end
    end
    return nothing
end


function kernel2(f::CUDA.CuDeviceArray{T, 2}, ftmp, TT, region) where T
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    # axis = 1
    N = length(region)
    Naxis = size(region)[1]   # Naxis = size(region)[axis]
    for k=id:stride:N
        i = region[k][1]   # i = region[k][axis]
        j = region[k][2]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m, j]
        end
    end
    return nothing
end


function kernel3(f::CUDA.CuDeviceArray, ftmp, J, RV, region)
    id = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    stride = CUDA.blockDim().x * CUDA.gridDim().x
    # axis = 1
    N = length(region)
    for k=id:stride:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds f[k] = ftmp[k] * J[i] / RV
    end
    return nothing
end


end
