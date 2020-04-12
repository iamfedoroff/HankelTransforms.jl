for p in [0, 1, 4]
    R = 3.0
    N1 = 256
    N2 = 512

    v = HankelTransforms.htfreq(R, N1, p)
    r = HankelTransforms.htcoord(R, N1, p)
    t = range(zero(R), R, length=N2)

    f1 = zeros(typeof(R), (N1, N2))
    f2th = zeros(typeof(R), (N1, N2))
    for i=1:N1
    for j=1:N2
        f1[i, j] = mysinc(r[i]) * exp(-(t[j] - R / 2)^2)
        f2th[i, j] = mysinc_spectrum(v[i], p) * exp(-(t[j] - R / 2)^2)
    end
    end

    plan = HankelTransforms.plan(R, f1, p)
    f2 = HankelTransforms.dht(f1, plan)
    f3 = HankelTransforms.idht(f2, plan)

    err = 20 * log10.(abs.(f2th .- f2) / maximum(abs.(f2)))

    @test maximum(err) < -10
    @test isapprox(f1, f3)

    @allocated HankelTransforms.dht!(f1, plan)
    @test (@allocated HankelTransforms.dht!(f1, plan)) == 0

    @allocated HankelTransforms.idht!(f2, plan)
    @test (@allocated HankelTransforms.idht!(f2, plan)) == 0
end
