module HankelTransforms

import GSL
import LinearAlgebra

export plan, htcoord, htfreq, dht!, dht, idht!, idht

const htAbstractArray{T} = Union{AbstractArray{T}, AbstractArray{Complex{T}}}


struct Plan{
    I<:Int,
    T<:AbstractFloat,
    UA<:AbstractArray{T},
    UB<:AbstractArray{T},
    UC<:htAbstractArray{T},
}
    N :: I
    R :: T
    V :: T
    J :: UA
    TT :: UB
    ftmp :: UC
end


function plan(
    R::T, A::U, p::I=0,
) where {T<:AbstractFloat, U<:htAbstractArray{T}, I<:Int}
    dims = size(A)
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

    ftmp = zero(A)

    return Plan(N, R, V, J, TT, ftmp)
end


"""
Compute the spatial coordinates for Hankel transform.
"""
function htcoord(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = @. GSL.sf_bessel_zero_Jnu(p, 1:N)
    aNp1 = GSL.sf_bessel_zero_Jnu(p, N + 1)
    V = aNp1 / (2 * pi * R)
    @. a = a / (2 * pi * V)   # resuse the same array to avoid allocations
    return a
end


"""
Compute the spatial frequencies (ordinary, not angular) for Hankel transform.
"""
function htfreq(R::T, N::I, p::I=0) where {T<:AbstractFloat, I<:Int}
    a = @. GSL.sf_bessel_zero_Jnu(p, 1:N)
    @. a = a / (2 * pi * R)   # resuse the same array to avoid allocations
    return a
end


"""
Compute (in place) forward discrete Hankel transform.
"""
function dht!(f::UC, plan::Plan{I, T, UA, UB, UC}) where {I, T, UA, UB, UC}
    @. f = f * plan.R / plan.J
    LinearAlgebra.mul!(plan.ftmp, plan.TT, f)
    @. f = plan.ftmp * plan.J / plan.V
    return nothing
end


"""
Compute (out of place) forward discrete Hankel transform.
"""
function dht(f1::UC, plan::Plan{I, T, UA, UB, UC}) where {I, T, UA, UB, UC}
    f2 = copy(f1)
    dht!(f2, plan)
    return f2
end


"""
Compute (in place) backward discrete Hankel transform.
"""
function idht!(f::UC, plan::Plan{I, T, UA, UB, UC}) where {I, T, UA, UB, UC}
    @. f = f * plan.V / plan.J
    LinearAlgebra.mul!(plan.ftmp, plan.TT, f)
    @. f = plan.ftmp * plan.J / plan.R
    return nothing
end


"""
Compute (out of place) backward discrete Hankel transform.
"""
function idht(f2::UC, plan::Plan{I, T, UA, UB, UC}) where {I, T, UA, UB, UC}
    f1 = copy(f2)
    idht!(f1, plan)
    return f1
end


end
