module HankelTransforms

import GSL
import LinearAlgebra

# Import GPU packages:
import CUDAapi

isongpu = function(T)
    return false
end

cuconvert = function(F)
    return F
end

if CUDAapi.has_cuda()   # check that CUDA is installed
if CUDAapi.has_cuda_gpu()   # check that GPU is active
    try
        import CuArrays   # we have CUDA, so this should not fail
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

    catch ex
        # something is wrong with the user's set-up (or there's a bug in CuArrays)
        @warn "CUDA is installed, but CuArrays.jl fails to load"
            exception = (ex, catch_backtrace())

    end
end
end


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
Compute (in place) forward discrete Hankel transform.
"""
function dht!(
    f::UF, plan::Plan{IOG, I, T, UJ, UT, UF},
) where {IOG, I, T, UJ, UT, UF}
    @. f = f * plan.R / plan.J
    LinearAlgebra.mul!(plan.ftmp, plan.TT, f)
    @. f = plan.ftmp * plan.J / plan.V
    return nothing
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
Compute (in place) backward discrete Hankel transform.
"""
function idht!(
    f::UF, plan::Plan{IOG, I, T, UJ, UT, UF},
) where {IOG, I, T, UJ, UT, UF}
    @. f = f * plan.V / plan.J
    LinearAlgebra.mul!(plan.ftmp, plan.TT, f)
    @. f = plan.ftmp * plan.J / plan.R
    return nothing
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


end
