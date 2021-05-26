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


struct DHTPlan{A<:AbstractArray, T<:AbstractFloat, C<:CartesianIndices} <: Plan
    N :: Int
    region :: C
    R :: T
    V :: T
    J :: Vector{T}
    TT :: Matrix{T}
    ftmp :: A
end


struct CuDHTPlan{A<:CuArray, T<:AbstractFloat, C<:CartesianIndices} <: Plan
    N :: Int
    region :: C
    R :: T
    V :: T
    J :: CuVector{T}
    TT :: CuMatrix{T}
    ftmp :: A
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
    save::Bool=false,
    fname::String="dht.jld2",
)
    dims = size(region)
    N = dims[1]

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

    ftmp = zeros(eltype(F), dims)

    TF = real(eltype(F))
    TC = typeof(region)

    if typeof(F) <: CuArray
        ftmp = CuArray(ftmp)
        TA = typeof(ftmp)
        plan = CuDHTPlan{TA, TF, TC}(N, region, R, V, J, TT, ftmp)
    else
        TA = typeof(ftmp)
        plan = DHTPlan{TA, TF, TC}(N, region, R, V, J, TT, ftmp)
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
    kernel1(f, plan.J, plan.R, plan.region)
    kernel2(f, plan.ftmp, plan.TT, plan.region)
    kernel3(f, plan.ftmp, plan.J, plan.V, plan.region)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on CPU.
"""
function idht!(f::T, plan::DHTPlan{T}) where T
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


function kernel2(f::Vector, ftmp, TT, region)
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


function kernel2(f::Matrix, ftmp, TT, region)
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
function dht!(f::T, plan::CuDHTPlan{T}) where T
    N = length(plan.region)
    @krun N kernel1(f, plan.J, plan.R, plan.region)
    @krun N kernel2(f, plan.ftmp, plan.TT, plan.region)
    @krun N kernel3(f, plan.ftmp, plan.J, plan.V, plan.region)
    return nothing
end


"""
Compute (in place) backward discrete Hankel transform on GPU.
"""
function idht!(f::T, plan::CuDHTPlan{T}) where T
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


function kernel2(f::CuDeviceVector, ftmp, TT, region)
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


function kernel2(f::CuDeviceMatrix, ftmp, TT, region)
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
