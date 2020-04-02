module HankelTransforms

import GSL
import LinearAlgebra


# Import GPU packages ----------------------------------------------------------
import CuArrays
import CUDAapi
import CUDAdrv
import CUDAnative

isongpu = function(T)
    return false
end

cuconvert = function(F)
    return F
end

# check that CUDA is installed and GPU is active:
if CUDAapi.has_cuda_gpu()
    CuArrays.allowscalar(false)   # disable slow fallback methods

    global isongpu = function(T)
        if T <: CuArrays.CuArray
            IOG = true
        else
            IOG = false
        end
        return IOG
    end

    global cuconvert = function(F)
        return CuArrays.CuArray(F)
    end
end
# ------------------------------------------------------------------------------


export plan, htcoord, htfreq, dht!, dht, idht!, idht

const htAbstractArray{T} = Union{AbstractArray{T}, AbstractArray{Complex{T}}}


struct Plan{
    IOG,
    I<:Int,
    T<:AbstractFloat,
    UJ<:AbstractArray{T},
    UT<:AbstractArray{T},
    UF<:htAbstractArray{T},
}
    N :: I
    R :: T
    V :: T
    J :: UJ
    TT :: UT
    ftmp :: UF
end


function plan(
    R::T, F::UF, p::I=0,
) where {T<:AbstractFloat, UF<:htAbstractArray{T}, I<:Int}
    dims = size(F)
    N = dims[1]

    a = zeros(T, N)
    J = zeros(T, N)
    TT = zeros(T, (N, N))

    @. a = GSL.sf_bessel_zero_Jnu(p, 1:N)
    aNp1::T = GSL.sf_bessel_zero_Jnu(p, N + 1)

    V::T = aNp1 / (2 * pi * R)
    @. J = abs(GSL.sf_bessel_Jn(p + 1, a)) / R

    S::T = 2 * pi * R * V

    for i=1:N
    for j=1:N
        TT[i, j] = 2 * GSL.sf_bessel_Jn(p, a[i] * a[j] / S) /
                   abs(GSL.sf_bessel_Jn(p + 1, a[i])) /
                   abs(GSL.sf_bessel_Jn(p + 1, a[j])) / S
    end
    end

    ftmp = zero(F)

    IOG = isongpu(UF)
    if IOG
        J = cuconvert(J)
        TT = cuconvert(TT)
    end
    UJ = typeof(J)
    UT = typeof(TT)

    return Plan{IOG, I, T, UJ, UT, UF}(N, R, V, J, TT, ftmp)
end


"""
Compute the spatial coordinates for Hankel transform.
"""
function htcoord(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = zeros(T, N)
    @. a = GSL.sf_bessel_zero_Jnu(p, 1:N)
    aNp1::T = GSL.sf_bessel_zero_Jnu(p, N + 1)
    V::T = aNp1 / (2 * pi * R)
    @. a = a / (2 * pi * V)   # resuse the same array to avoid allocations
    return a
end


"""
Compute the spatial frequencies (ordinary, not angular) for Hankel transform.
"""
function htfreq(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = zeros(T, N)
    @. a = GSL.sf_bessel_zero_Jnu(p, 1:N)
    @. a = a / (2 * pi * R)   # resuse the same array to avoid allocations
    return a
end


"""
Compute (out of place) forward discrete Hankel transform.
"""
function dht(
    f::UF, plan::Plan{IOG, I, T, UJ, UT, UF},
) where {IOG, I, T, UJ, UT, UF}
    ftmp = copy(f)
    dht!(ftmp, plan)
    return ftmp
end


"""
Compute (out of place) backward discrete Hankel transform.
"""
function idht(
    f::UF, plan::Plan{IOG, I, T, UJ, UT, UF},
) where {IOG, I, T, UJ, UT, UF}
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
    f::UF, plan::Plan{false, I, T, UJ, UT, UF},
) where {I, T, UJ, UT, UF}
    kernel1(f, plan.J, plan.R)
    kernel2(f, plan.ftmp, plan.TT)
    kernel3(f, plan.ftmp, plan.J, plan.V)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on CPU.
"""
function idht!(
    f::UF, plan::Plan{false, I, T, UJ, UT, UF},
) where {I, T, UJ, UT, UF}
    kernel1(f, plan.J, plan.V)
    kernel2(f, plan.ftmp, plan.TT)
    kernel3(f, plan.ftmp, plan.J, plan.R)
    return nothing
end


function kernel1(f, J, RV)
    # axis = 1
    N = length(f)
    cartesian = CartesianIndices(f)
    for k=1:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
        @inbounds f[k] = f[k] * RV / J[i]
    end
    return nothing
end


function kernel2(f::AbstractArray{T, 1}, ftmp, TT) where T
    # axis = 1
    N = length(f)
    dims = size(f)
    Naxis = dims[1]   # Naxis = dims[axis]
    cartesian = CartesianIndices(f)
    for k=1:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m]
        end
    end
    return nothing
end


function kernel2(f::AbstractArray{T, 2}, ftmp, TT) where T
    # axis = 1
    N = length(f)
    dims = size(f)
    Naxis = dims[1]   # Naxis = dims[axis]
    cartesian = CartesianIndices(f)
    for k=1:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
        j = cartesian[k][2]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m, j]
        end
    end
    return nothing
end


function kernel3(f, ftmp, J, RV)
    # axis = 1
    N = length(f)
    cartesian = CartesianIndices(f)
    for k=1:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
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
    f::UF, plan::Plan{true, I, T, UJ, UT, UF},
) where {I, T, UJ, UT, UF}
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(
        CUDAnative.CuDevice(0), CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    )
    N = length(f)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    CUDAnative.@cuda blocks=nbl threads=nth kernel1(f, plan.J, plan.R)
    CUDAnative.@cuda blocks=nbl threads=nth kernel2(f, plan.ftmp, plan.TT)
    CUDAnative.@cuda blocks=nbl threads=nth kernel3(f, plan.ftmp, plan.J, plan.V)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on GPU.
"""
function idht!(
    f::UF, plan::Plan{true, I, T, UJ, UT, UF},
) where {I, T, UJ, UT, UF}
    MAX_THREADS_PER_BLOCK = CUDAdrv.attribute(
        CUDAnative.CuDevice(0), CUDAdrv.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    )
    N = length(f)
    nth = min(N, MAX_THREADS_PER_BLOCK)
    nbl = cld(N, nth)
    CUDAnative.@cuda blocks=nbl threads=nth kernel1(f, plan.J, plan.V)
    CUDAnative.@cuda blocks=nbl threads=nth kernel2(f, plan.ftmp, plan.TT)
    CUDAnative.@cuda blocks=nbl threads=nth kernel3(f, plan.ftmp, plan.J, plan.R)
    return nothing
end


function kernel1(f::CUDAnative.CuDeviceArray, J, RV)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    # axis = 1
    N = length(f)
    cartesian = CartesianIndices(f)
    for k=id:stride:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
        @inbounds f[k] = f[k] * RV / J[i]
    end
    return nothing
end


function kernel2(f::CUDAnative.CuDeviceArray{T, 1}, ftmp, TT) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    # axis = 1
    N = length(f)
    dims = size(f)
    Naxis = dims[1]   # Naxis = dims[axis]
    cartesian = CartesianIndices(f)
    for k=id:stride:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m]
        end
    end
    return nothing
end


function kernel2(f::CUDAnative.CuDeviceArray{T, 2}, ftmp, TT) where T
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    # axis = 1
    N = length(f)
    dims = size(f)
    Naxis = dims[1]   # Naxis = dims[axis]
    cartesian = CartesianIndices(f)
    for k=id:stride:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
        j = cartesian[k][2]
        @inbounds ftmp[k] = 0
        for m=1:Naxis
            @inbounds ftmp[k] = ftmp[k] + TT[i, m] * f[m, j]
        end
    end
    return nothing
end


function kernel3(f::CUDAnative.CuDeviceArray, ftmp, J, RV)
    id = (CUDAnative.blockIdx().x - 1) * CUDAnative.blockDim().x +
         CUDAnative.threadIdx().x
    stride = CUDAnative.blockDim().x * CUDAnative.gridDim().x
    # axis = 1
    N = length(f)
    cartesian = CartesianIndices(f)
    for k=id:stride:N
        i = cartesian[k][1]   # i = cartesian[k][axis]
        @inbounds f[k] = ftmp[k] * J[i] / RV
    end
    return nothing
end


end
