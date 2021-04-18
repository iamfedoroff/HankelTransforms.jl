module HankelTransforms

import Base: *, \
import JLD2
import LinearAlgebra


# CPU / GPU specific functions -------------------------------------------------
import CUDA


function isongpu(T)
    if T <: CUDA.CuArray
        IOG = true
    else
        IOG = false
    end
    return IOG
end


macro krun(ex...)
    N = ex[1]
    call = ex[2]

    args = call.args[2:end]

    @gensym kernel config threads blocks
    code = quote
        local $kernel = CUDA.@cuda launch=false $call
        local $config = CUDA.launch_configuration($kernel.fun)
        local $threads = min($N, $config.threads)
        local $blocks = cld($N, $threads)
        $kernel($(args...); threads=$threads, blocks=$blocks)
    end

    return esc(code)
end


# Bessel functions -------------------------------------------------------------
# GSL is the Julia wrapper for the GNU Scientific Library (GSL) and does not
# work under Windows.
# import GSL
# bessel(p, x) = GSL.sf_bessel_Jn(p, x)
# bessel_zero(p, n) = GSL.sf_bessel_zero_Jnu(p, n)

import SpecialFunctions
bessel(p, x) = SpecialFunctions.besselj(p, x)

# import FunctionZeros
# bessel_zero(p, n) = FunctionZeros.besselj_zero(p, n)

# Since FunctionZeros package is too slow in updating of versions for
# SpecialFunctions, I copy all necessary functions here:

import Roots

# Asymptotic formula for the n'th zero of Bessel J function of order nu
besselj_zero_asymptotic(nu, n) = pi * (n - 1 + nu / 2 + 3//4)

# Use the asymptotic values as starting values. These find the correct zeros
# even for n = 1,2,... Order 0 is 6 times slower and 50-100 times less accurate
# than higher orders, with other parameters constant.
besselj_zero(nu, n; order=2) =
    Roots.fzero(
        (x) -> SpecialFunctions.besselj(nu, x),
        besselj_zero_asymptotic(nu, n); order=order
    )

bessel_zero(p, n) = besselj_zero(p, n)
# ------------------------------------------------------------------------------


export plan, htcoord, htfreq, dht!, dht, idht!, idht

const htAbstractArray{T} = Union{AbstractArray{T}, AbstractArray{Complex{T}}}


struct Plan{
    IOG,
    CI<:CartesianIndices,
    T<:AbstractFloat,
    UJ<:AbstractArray{T},
    UT<:AbstractArray{T},
    UF<:htAbstractArray{T},
}
    N :: Int
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

    ftmp = zeros(T, dims)

    IOG = isongpu(UF)
    if IOG
        J = CUDA.CuArray(J)
        TT = CUDA.CuArray(TT)
        ftmp = CUDA.CuArray(ftmp)
    end

    CI = typeof(region)
    UJ = typeof(J)
    UT = typeof(TT)
    plan = Plan{IOG, CI, T, UJ, UT, UF}(N, region, R, V, J, TT, ftmp)

    if save
        JLD2.@save fname plan
    end

    return plan
end


function plan(fname::String)
    plan = nothing
    JLD2.@load fname plan
    return plan
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
    f::UF, plan::Plan{IOG, CI, T, UJ, UT, UF},
) where {IOG, CI, T, UJ, UT, UF}
    ftmp = copy(f)
    dht!(ftmp, plan)
    return ftmp
end


"""
Compute (out of place) backward discrete Hankel transform.
"""
function idht(
    f::UF, plan::Plan{IOG, CI, T, UJ, UT, UF},
) where {IOG, CI, T, UJ, UT, UF}
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
    f::UF, plan::Plan{false, CI, T, UJ, UT, UF},
) where {CI, T, UJ, UT, UF}
    kernel1(f, plan.J, plan.R, plan.region)
    kernel2(f, plan.ftmp, plan.TT, plan.region)
    kernel3(f, plan.ftmp, plan.J, plan.V, plan.region)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on CPU.
"""
function idht!(
    f::UF, plan::Plan{false, CI, T, UJ, UT, UF},
) where {CI, T, UJ, UT, UF}
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
    f::UF, plan::Plan{true, CI, T, UJ, UT, UF},
) where {CI, T, UJ, UT, UF}
    N = length(plan.region)
    @krun N kernel1(f, plan.J, plan.R, plan.region)
    @krun N kernel2(f, plan.ftmp, plan.TT, plan.region)
    @krun N kernel3(f, plan.ftmp, plan.J, plan.V, plan.region)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on GPU.
"""
function idht!(
    f::UF, plan::Plan{true, CI, T, UJ, UT, UF},
) where {CI, T, UJ, UT, UF}
    N = length(plan.region)
    @krun N kernel1(f, plan.J, plan.V, plan.region)
    @krun N kernel2(f, plan.ftmp, plan.TT, plan.region)
    @krun N kernel3(f, plan.ftmp, plan.J, plan.R, plan.region)
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
