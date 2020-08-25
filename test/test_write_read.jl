let
    R = 3.0
    N = 256

    r = HankelTransforms.htcoord(R, N)
    f = @. mysinc(r)

    planw = HankelTransforms.plan(R, f, save=true)
    planr = HankelTransforms.plan("dht.jld2")
    rm("dht.jld2")

    @test fieldnames(typeof(planw)) == fieldnames(typeof(planr))
    for v in fieldnames(typeof(planw))
        @test getfield(planw, v) == getfield(planr, v)
    end
end
