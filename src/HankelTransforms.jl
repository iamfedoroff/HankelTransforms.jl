module HankelTransforms

import Base: *, \
import CUDA: CuArray, CuDeviceArray, @cuda, launch_configuration,
             threadIdx, blockIdx, blockDim, gridDim
import JLD2: @save, @load
import Roots: fzero
import SpecialFunctions: besselj

export plan, htcoord, htfreq, dht!, dht, idht!, idht


# Adopted from FunctionZeros.jl
# Asymptotic formula for the n'th zero of Bessel J function of order nu
besselj_zero_asymptotic(nu, n) = pi * (n - 1 + nu / 2 + 3//4)


# Adopted from FunctionZeros.jl
# Use the asymptotic values as starting values. These find the correct zeros
# even for n = 1,2,... Order 0 is 6 times slower and 50-100 times less accurate
# than higher orders, with other parameters constant.
besselj_zero(nu, n; order=2) =
    fzero((x) -> besselj(nu, x), besselj_zero_asymptotic(nu, n); order=order)


macro krun(ex...)
    N = ex[1]
    call = ex[2]

    args = call.args[2:end]

    @gensym kernel config threads blocks
    code = quote
        local $kernel = @cuda launch=false $call
        local $config = launch_configuration($kernel.fun)
        local $threads = min($N, $config.threads)
        local $blocks = cld($N, $threads)
        $kernel($(args...); threads=$threads, blocks=$blocks)
    end

    return esc(code)
end


const htAbstractArray{T} = Union{AbstractArray{T}, AbstractArray{Complex{T}}}


abstract type Plan end


struct DHTPlan{
    CI<:CartesianIndices,
    T<:AbstractFloat,
    UJ<:AbstractArray{T},
    UT<:AbstractArray{T},
    UF<:htAbstractArray{T},
} <: Plan
    N :: Int
    region :: CI
    R :: T
    V :: T
    J :: UJ
    TT :: UT
    ftmp :: UF
end


struct CuDHTPlan{
    CI<:CartesianIndices,
    T<:AbstractFloat,
    UJ<:AbstractArray{T},
    UT<:AbstractArray{T},
    UF<:htAbstractArray{T},
} <: Plan
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

    @. a = besselj_zero(p, 1:N)
    aNp1::T = besselj_zero(p, N + 1)

    V::T = aNp1 / (2 * pi * R)
    @. J = abs(besselj(p + 1, a)) / R

    S::T = 2 * pi * R * V

    for j=1:N
    for i=1:N
        TT[i, j] = 2 * besselj(p, a[i] * a[j] / S) /
                   abs(besselj(p + 1, a[i])) /
                   abs(besselj(p + 1, a[j])) / S
    end
    end

    ftmp = zeros(T, dims)

    CI = typeof(region)

    if typeof(F) <: CuArray
        J = CuArray(J)
        TT = CuArray(TT)
        ftmp = CuArray(ftmp)

        UJ = typeof(J)
        UT = typeof(TT)

        plan = CuDHTPlan{CI, T, UJ, UT, UF}(N, region, R, V, J, TT, ftmp)
    else
        UJ = typeof(J)
        UT = typeof(TT)

        plan = DHTPlan{CI, T, UJ, UT, UF}(N, region, R, V, J, TT, ftmp)
    end

    if save
        @save fname plan
    end

    return plan
end


function plan(fname::String)
    plan = nothing
    @load fname plan
    return plan
end


"""
Compute the spatial coordinates for Hankel transform.
"""
function htcoord(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = zeros(T, N)
    @. a = besselj_zero(p, 1:N)
    aNp1::T = besselj_zero(p, N + 1)
    V::T = aNp1 / (2 * pi * R)
    @. a = a / (2 * pi * V)   # resuse the same array to avoid allocations
    return a
end


"""
Compute the spatial frequencies (ordinary, not angular) for Hankel transform.
"""
function htfreq(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = zeros(T, N)
    @. a = besselj_zero(p, 1:N)
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
function dht(f::AbstractArray, plan::Plan)
    ftmp = copy(f)
    dht!(ftmp, plan)
    return ftmp
end


"""
Compute (out of place) backward discrete Hankel transform.
"""
function idht(f::AbstractArray, plan::Plan)
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
    f::UF, plan::DHTPlan{CI, T, UJ, UT, UF},
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
    f::UF, plan::DHTPlan{CI, T, UJ, UT, UF},
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
    f::UF, plan::CuDHTPlan{CI, T, UJ, UT, UF},
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
    f::UF, plan::CuDHTPlan{CI, T, UJ, UT, UF},
) where {CI, T, UJ, UT, UF}
    N = length(plan.region)
    @krun N kernel1(f, plan.J, plan.V, plan.region)
    @krun N kernel2(f, plan.ftmp, plan.TT, plan.region)
    @krun N kernel3(f, plan.ftmp, plan.J, plan.R, plan.region)
    return nothing
end


function kernel1(f::CuDeviceArray, J, RV, region)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    # axis = 1
    N = length(region)
    for k=id:stride:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds f[k] = f[k] * RV / J[i]
    end
    return nothing
end


function kernel2(f::CuDeviceArray{T, 1}, ftmp, TT, region) where T
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
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


function kernel2(f::CuDeviceArray{T, 2}, ftmp, TT, region) where T
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
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


function kernel3(f::CuDeviceArray, ftmp, J, RV, region)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    # axis = 1
    N = length(region)
    for k=id:stride:N
        i = region[k][1]   # i = region[k][axis]
        @inbounds f[k] = ftmp[k] * J[i] / RV
    end
    return nothing
end


end
