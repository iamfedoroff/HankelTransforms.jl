for p in [0, 1, 4]
    R = 3f0
    N = 1024

    v = HankelTransforms.htfreq(R, N, p)
    r = HankelTransforms.htcoord(R, N, p)
    f1 = @. mysinc(r)
    f2th = @. mysinc_spectrum(v, p)

    f1 = Array{typeof(R)}(f1)
    f2th = Array{typeof(R)}(f2th)

    f_gpu = CUDA.CuArray(f1)
    plan_gpu = HankelTransforms.plan(R, f_gpu, p)

    HankelTransforms.dht!(f_gpu, plan_gpu)
    f2 = CUDA.collect(f_gpu)

    HankelTransforms.idht!(f_gpu, plan_gpu)
    f3 = CUDA.collect(f_gpu)

    err = 20 * log10.(abs.(f2th .- f2) / maximum(abs.(f2)))

    @test maximum(err) < -10
    @test isapprox(f1, f3)
end
