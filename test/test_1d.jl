for p in [0, 1, 4]
    R = 3.0
    N = 256

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
