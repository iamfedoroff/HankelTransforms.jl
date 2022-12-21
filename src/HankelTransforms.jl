module HankelTransforms

import Base: *, \
import CUDA: CuArray, CuVector, CuMatrix, CuDeviceArray, CuDeviceVector,
             CuDeviceMatrix, @cuda, launch_configuration, threadIdx, blockIdx,
             blockDim, gridDim
import JLD2: @save, @load
import Roots: fzero
import SpecialFunctions: besselj

export plan_dht, dhtcoord, dhtfreq, dht!, dht, idht!, idht


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
        local $blocks = min($config.blocks, cld($N, $threads))
        $kernel($(args...); threads=$threads, blocks=$blocks)
    end

    return esc(code)
end


abstract type Plan end


struct DHTPlan{TA, T, TIpre, TIpos, TItot} <: Plan
    N :: Int
    R :: T
    V :: T
    J :: Vector{T}
    TT :: Matrix{T}
    Atmp :: TA
    Ipre :: TIpre
    Ipos :: TIpos
    Itot :: TItot
end


struct CuDHTPlan{TA, T, TIpre, TIpos, TItot} <: Plan
    N :: Int
    R :: T
    V :: T
    J :: CuVector{T}
    TT :: CuMatrix{T}
    Atmp :: TA
    Ipre :: TIpre
    Ipos :: TIpos
    Itot :: TItot
end


function plan_dht(
    R::Real,
    A::AbstractArray;
    p::Int=0,
    dim::Int=1,
    region::Tuple=size(A),
    save::Bool=false,
    fname::String="plan_dht.jld2",
)
    N = region[dim]

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

    Atmp = zeros(eltype(A), size(CartesianIndices(region)))

    Ipre = CartesianIndices(region[1:dim-1])
    Ipos = CartesianIndices(region[dim+1:end])
    Itot = CartesianIndices((length(Ipre), N, length(Ipos)))

    TF = real(eltype(A))
    TIpre = typeof(Ipre)
    TIpos = typeof(Ipos)
    TItot = typeof(Itot)

    if typeof(A) <: CuArray
        Atmp = CuArray(Atmp)
        TA = typeof(Atmp)

        plan = CuDHTPlan{TA, TF, TIpre, TIpos, TItot}(
            N, R, V, J, TT, Atmp, Ipre, Ipos, Itot,
        )
    else
        TA = typeof(Atmp)

        plan = DHTPlan{TA, TF, TIpre, TIpos, TItot}(
            N, R, V, J, TT, Atmp, Ipre, Ipos, Itot,
        )
    end

    if save
        @save fname plan
    end

    return plan
end


function plan_dht(fname::String)
    plan = nothing
    @load fname plan
    return plan
end


"""
Compute the spatial coordinates for Hankel transform.
"""
function dhtcoord(R::Real, N::Int; p::Int=0)
    a = @. besselj_zero(p, 1:N)
    aNp1 = besselj_zero(p, N + 1)
    V = aNp1 / (2 * pi * R)
    return @. a / (2 * pi * V)
end


"""
Compute the spatial frequencies (ordinary, not angular) for Hankel transform.
"""
function dhtfreq(R::Real, N::Int; p::Int=0)
    a = @. besselj_zero(p, 1:N)
    return @. a / (2 * pi * R)
end


# ******************************************************************************
# AbstractFFTs API
# ******************************************************************************
function *(p::Plan, A::AbstractArray)
    dht!(A, p)
    return nothing
end


function \(p::Plan, A::AbstractArray)
    idht!(A, p)
    return nothing
end


# ******************************************************************************
# Out of place functions
# ******************************************************************************
"""
Compute (out of place) forward discrete Hankel transform.
"""
function dht(A::AbstractArray, p::Plan)
    Atmp = copy(A)
    dht!(Atmp, p)
    return Atmp
end


"""
Compute (out of place) backward discrete Hankel transform.
"""
function idht(A::AbstractArray, p::Plan)
    Atmp = copy(A)
    idht!(Atmp, p)
    return Atmp
end


# ******************************************************************************
# CPU functions
# ******************************************************************************
"""
Compute (in place) forward discrete Hankel transform on CPU.
"""
function dht!(A::T, p::DHTPlan{T}) where T
    kernel1(A, p.J, p.R, p.Ipre, p.Ipos, p.Itot)
    kernel2(p.Atmp, p.TT, A, p.Ipre, p.Ipos, p.Itot, p.N)
    kernel3(A, p.Atmp, p.J, p.V, p.Ipre, p.Ipos, p.Itot)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on CPU.
"""
function idht!(A::T, p::DHTPlan{T}) where T
    kernel1(A, p.J, p.V, p.Ipre, p.Ipos, p.Itot)
    kernel2(p.Atmp, p.TT, A, p.Ipre, p.Ipos, p.Itot, p.N)
    kernel3(A, p.Atmp, p.J, p.R, p.Ipre, p.Ipos, p.Itot)
    return nothing
end


function kernel1(A, J, RV, Ipre, Ipos, Itot)
    for k=1:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds A[ipre, idim, ipos] = A[ipre, idim, ipos] * RV / J[idim]
    end
    return nothing
end


function kernel2(Atmp, TT, A, Ipre, Ipos, Itot, N)
    for k=1:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        res = zero(eltype(Atmp))
        for m=1:N
            @inbounds res = res + TT[idim, m] * A[ipre, m, ipos]
        end
        @inbounds Atmp[k] = res
    end
    return nothing
end


function kernel3(A, Atmp, J, RV, Ipre, Ipos, Itot)
    for k=1:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds A[ipre, idim, ipos] = Atmp[k] * J[idim] / RV
    end
    return nothing
end


# ******************************************************************************
# GPU functions
# ******************************************************************************
"""
Compute (in place) forward discrete Hankel transform on GPU.
"""
function dht!(A::T, p::CuDHTPlan{T}) where T
    N = length(p.Itot)
    @krun N kernel1(A, p.J, p.R, p.Ipre, p.Ipos, p.Itot)
    @krun N kernel2(p.Atmp, p.TT, A, p.Ipre, p.Ipos, p.Itot, p.N)
    @krun N kernel3(A, p.Atmp, p.J, p.V, p.Ipre, p.Ipos, p.Itot)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on GPU.
"""
function idht!(A::T, p::CuDHTPlan{T}) where T
    N = length(p.Itot)
    @krun N kernel1(A, p.J, p.V, p.Ipre, p.Ipos, p.Itot)
    @krun N kernel2(p.Atmp, p.TT, A, p.Ipre, p.Ipos, p.Itot, p.N)
    @krun N kernel3(A, p.Atmp, p.J, p.R, p.Ipre, p.Ipos, p.Itot)
    return nothing
end


function kernel1(A::CuDeviceArray, J, RV, Ipre, Ipos, Itot)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for k=id:stride:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds A[ipre, idim, ipos] = A[ipre, idim, ipos] * RV / J[idim]
    end
    return nothing
end


function kernel2(Atmp, TT, A::CuDeviceArray, Ipre, Ipos, Itot, N)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for k=id:stride:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        res = zero(eltype(Atmp))
        for m=1:N
            @inbounds res = res + TT[idim, m] * A[ipre, m, ipos]
        end
        @inbounds Atmp[k] = res
    end
    return nothing
end


function kernel3(A::CuDeviceArray, Atmp, J, RV, Ipre, Ipos, Itot)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for k=id:stride:length(Itot)
        @inbounds ipre = Ipre[Itot[k][1]]
        @inbounds idim = Itot[k][2]
        @inbounds ipos = Ipos[Itot[k][3]]
        @inbounds A[ipre, idim, ipos] = Atmp[k] * J[idim] / RV
    end
    return nothing
end


end
