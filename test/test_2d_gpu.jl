for p in [0, 1, 4]
    R = 3f0
    N1 = 256
    N2 = 2

    v = HankelTransforms.htfreq(R, N1, p)
    r = HankelTransforms.htcoord(R, N1, p)

    f1 = zeros(typeof(R), (N1, N2))
    f2th = zeros(typeof(R), (N1, N2))
    for i=1:N1
    for j=1:N2
        f1[i, j] = mysinc(r[i])
        f2th[i, j] = mysinc_spectrum(v[i], p)
    end
    end

    f_gpu = CuArrays.CuArray(f1)
    plan_gpu = HankelTransforms.plan(R, f_gpu, p)

    HankelTransforms.dht!(f_gpu, plan_gpu)
    f2 = CuArrays.collect(f_gpu)

    HankelTransforms.idht!(f_gpu, plan_gpu)
    f3 = CuArrays.collect(f_gpu)

    err = 20 * log10.(abs.(f2th .- f2) / maximum(abs.(f2)))

    @test maximum(err) < -10
    @test isapprox(f1, f3)
end
