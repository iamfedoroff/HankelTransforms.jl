module HankelTransforms

import Base: *, \
import CUDA: CuArray, CuVector, CuMatrix, CuDeviceArray, CuDeviceVector,
             CuDeviceMatrix, @cuda, launch_configuration, threadIdx, blockIdx,
             blockDim, gridDim
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


abstract type Plan end


struct DHTPlan{A, T, TIpre, TIpos, TItot} <: Plan
    N :: Int
    R :: T
    V :: T
    J :: Vector{T}
    TT :: Matrix{T}
    ftmp :: A
    Ipre :: TIpre
    Ipos :: TIpos
    Itot :: TItot
end


struct CuDHTPlan{A, T, TIpre, TIpos, TItot} <: Plan
    N :: Int
    R :: T
    V :: T
    J :: CuVector{T}
    TT :: CuMatrix{T}
    ftmp :: A
    Ipre :: TIpre
    Ipos :: TIpos
    Itot :: TItot
end


function plan(R::Real, F::AbstractArray, p::Int=0; kwargs...)
    region = CartesianIndices(F)
    return plan(R, F, region, p; kwargs...)
end


function plan(
    R::Real,
    F::AbstractArray,
    region::CartesianIndices,
    p::Int=0;
    dim::Int=1,
    save::Bool=false,
    fname::String="dht.jld2",
)
    # N = region[dim]
    N = size(region)[dim]

    Ipre = CartesianIndices(region.indices[1:dim-1])
    Ipos = CartesianIndices(region.indices[dim+1:end])
    Itot = CartesianIndices((length(Ipre), N, length(Ipos)))

    a = @. besselj_zero(p, 1:N)
    aNp1 = besselj_zero(p, N + 1)

    V = aNp1 / (2 * pi * R)
    J = @. abs(besselj(p + 1, a)) / R

    S = 2 * pi * R * V
    TT = zeros((N, N))
    for j=1:N
    for i=1:N
        TT[i, j] = 2 * besselj(p, a[i] * a[j] / S) /
                   abs(besselj(p + 1, a[i])) /
                   abs(besselj(p + 1, a[j])) / S
    end
    end

    ftmp = zeros(eltype(F), size(region))

    TF = real(eltype(F))
    TIpre = typeof(Ipre)
    TIpos = typeof(Ipos)
    TItot = typeof(Itot)

    if typeof(F) <: CuArray
        ftmp = CuArray(ftmp)
        TA = typeof(ftmp)

        plan = CuDHTPlan{TA, TF, TIpre, TIpos, TItot}(
            N, R, V, J, TT, ftmp, Ipre, Ipos, Itot,
        )
    else
        TA = typeof(ftmp)

        plan = DHTPlan{TA, TF, TIpre, TIpos, TItot}(
            N, R, V, J, TT, ftmp, Ipre, Ipos, Itot,
        )
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
function htcoord(R::Real, N::Int, p::Int=0)
    a = @. besselj_zero(p, 1:N)
    aNp1 = besselj_zero(p, N + 1)
    V = aNp1 / (2 * pi * R)
    return @. a / (2 * pi * V)
end


"""
Compute the spatial frequencies (ordinary, not angular) for Hankel transform.
"""
function htfreq(R::Real, N::Int, p::Int=0)
    a = @. besselj_zero(p, 1:N)
    return @. a / (2 * pi * R)
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
function dht!(f::T, plan::DHTPlan{T}) where T
    kernel1(f, plan.J, plan.R, plan.Ipre, plan.Ipos, plan.Itot)
    kernel2(plan.ftmp, plan.TT, f, plan.Ipre, plan.Ipos, plan.Itot, plan.N)
    kernel3(f, plan.ftmp, plan.J, plan.V, plan.Ipre, plan.Ipos, plan.Itot)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on CPU.
"""
function idht!(f::T, plan::DHTPlan{T}) where T
    kernel1(f, plan.J, plan.V, plan.Ipre, plan.Ipos, plan.Itot)
    kernel2(plan.ftmp, plan.TT, f, plan.Ipre, plan.Ipos, plan.Itot, plan.N)
    kernel3(f, plan.ftmp, plan.J, plan.R, plan.Ipre, plan.Ipos, plan.Itot)
    return nothing
end


function kernel1(f, J, RV, Ipre, Ipos, Itot)
    for k=1:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds f[ipre, idim, ipos] = f[ipre, idim, ipos] * RV / J[idim]
    end
    return nothing
end


function kernel2(ftmp, TT, f, Ipre, Ipos, Itot, N)
    for k=1:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        res = zero(eltype(ftmp))
        for m=1:N
            @inbounds res = res + TT[idim, m] * f[ipre, m, ipos]
        end
        @inbounds ftmp[k] = res
    end
    return nothing
end


function kernel3(f, ftmp, J, RV, Ipre, Ipos, Itot)
    for k=1:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds f[ipre, idim, ipos] = ftmp[k] * J[idim] / RV
    end
    return nothing
end


# ******************************************************************************
# GPU functions
# ******************************************************************************
"""
Compute (in place) forward discrete Hankel transform on GPU.
"""
function dht!(f::T, plan::CuDHTPlan{T}) where T
    N = length(plan.Itot)
    @krun N kernel1(f, plan.J, plan.R, plan.Ipre, plan.Ipos, plan.Itot)
    @krun N kernel2(plan.ftmp, plan.TT, f, plan.Ipre, plan.Ipos, plan.Itot, plan.N)
    @krun N kernel3(f, plan.ftmp, plan.J, plan.V, plan.Ipre, plan.Ipos, plan.Itot)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on GPU.
"""
function idht!(f::T, plan::CuDHTPlan{T}) where T
    N = length(plan.Itot)
    @krun N kernel1(f, plan.J, plan.V, plan.Ipre, plan.Ipos, plan.Itot)
    @krun N kernel2(plan.ftmp, plan.TT, f, plan.Ipre, plan.Ipos, plan.Itot, plan.N)
    @krun N kernel3(f, plan.ftmp, plan.J, plan.R, plan.Ipre, plan.Ipos, plan.Itot)
    return nothing
end


function kernel1(f::CuDeviceArray, J, RV, Ipre, Ipos, Itot)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for k=id:stride:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds f[ipre, idim, ipos] = f[ipre, idim, ipos] * RV / J[idim]
    end
    return nothing
end


function kernel2(ftmp, TT, f::CuDeviceArray, Ipre, Ipos, Itot, N)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for k=id:stride:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        res = zero(eltype(ftmp))
        for m=1:N
            @inbounds res = res + TT[idim, m] * f[ipre, m, ipos]
        end
        @inbounds ftmp[k] = res
    end
    return nothing
end


function kernel3(f::CuDeviceArray, ftmp, J, RV, Ipre, Ipos, Itot)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for k=id:stride:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds f[ipre, idim, ipos] = ftmp[k] * J[idim] / RV
    end
    return nothing
end


end
