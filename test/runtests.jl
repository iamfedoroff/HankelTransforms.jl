import HankelTransforms
using Test

const gamma = 5.0
const R = 3.0
const N = 256


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


for p in [0, 1, 4]
    v = HankelTransforms.htfreq(R, N, p)
    r = HankelTransforms.htcoord(R, N, p)
    f1 = @. mysinc(r)
    f2th = @. mysinc_spectrum(v, p)

    plan = HankelTransforms.plan(R, f1, p)
    f2 = HankelTransforms.dht(f1, plan)
    f3 = HankelTransforms.idht(f2, plan)

    err = 20 * log10.(abs.(f2th .- f2) / maximum(abs.(f2)))

    @test maximum(err) < 10
    @test isapprox(f1, f3)

    @allocated HankelTransforms.dht!(f1, plan)
    @test (@allocated HankelTransforms.dht!(f1, plan)) == 0

    @allocated HankelTransforms.idht!(f2, plan)
    @test (@allocated HankelTransforms.idht!(f2, plan)) == 0
end
