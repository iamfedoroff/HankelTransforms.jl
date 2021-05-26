using CUDA
using HankelTransforms
using Test


function mysinc(r)
    gamma = 5
    return sin(2 * pi * gamma * r) / (2 * pi * gamma * r)
end


function mysinc_spectrum(v, p)
    gamma = 5
    if (v >= 0) & (v < gamma)
        f = v^p * cos(p * pi / 2) /
            (2 * pi * gamma * sqrt(gamma^2 - v^2) *
            (gamma + sqrt(gamma^2 - v^2))^p)
    elseif v > gamma
        f = sin(p * asin(gamma / v)) / (2 * pi * gamma * sqrt(v^2 - gamma^2))
    end
    return f
end


function test(R, p, s, dim; atype=Float64, cuda=false, region=nothing)
    N = s[dim]

    # r = htcoord(R, N; p=p)
    # v = htfreq(R, N; p=p)
    r = htcoord(R, N, p)
    v = htfreq(R, N, p)

    A1 = zeros(atype, s)
    A2th = zeros(atype, s)
    for I in CartesianIndices(A1)
        i = I[dim]
        A1[I] = mysinc(r[i])
        A2th[I] = mysinc_spectrum(v[i], p)
    end

    if cuda
        A1 = CuArray(A1)
        A2th = CuArray(A2th)
    end

    # ht = plan(R, A1; dim=dim, region=region, p=p)
    ht = plan(R, A1, p)

    A2 = copy(A1)
    dht!(A2, ht)

    A3 = copy(A2)
    idht!(A3, ht)

    if isnothing(region)
        err = 20 * log10.(abs.(A2th .- A2) / maximum(abs.(A2)))
        @test maximum(err) < -10
    end

    @test isapprox(A1, A3)

    if cuda
        CUDA.@allocated dht!(A2, ht)
        @test (CUDA.@allocated dht!(A2, ht)) == 0
        # @show (CUDA.@allocated dht!(A2, ht))

        CUDA.@allocated idht!(A3, ht)
        @test (CUDA.@allocated idht!(A3, ht)) == 0
        # @show (CUDA.@allocated idht!(A3, ht))
    else
        @allocated dht!(A2, ht)
        @test (@allocated dht!(A2, ht)) == 0
        # @show (@allocated dht!(A2, ht))

        @allocated idht!(A3, ht)
        @test (@allocated idht!(A3, ht)) == 0
        # @show (@allocated idht!(A3, ht))
    end

    return nothing
end


R = 3.0
N = 256


@testset "CPU" begin
    for p in [0, 1, 4]
        # 1D:
        test(R, p, (N, ), 1)

        # 2D:
        test(R, p, (N, 64), 1)
        # test(R, p, (64, N), 2)

        # 3D:
        # test(R, p, (N, 32, 64), 1)
        # test(R, p, (32, N, 64), 2)
        # test(R, p, (32, 64, N), 3)
    end

    # Different types:
    # 1D:
    test(R, 0, (N, ), 1; atype=Float32)
    test(R, 0, (N, ), 1; atype=Complex{Float32})
    test(R, 0, (N, ), 1; atype=Complex{Float64})

    # 2D:
    test(R, 0, (N, 64), 1; atype=Float32)
    test(R, 0, (N, 64), 1; atype=Complex{Float32})
    test(R, 0, (N, 64), 1; atype=Complex{Float64})

    # 3D:
    # test(R, 0, (N, 32, 64), 1; atype=Float32)
    # test(R, 0, (N, 32, 64), 1; atype=Complex{Float32})
    # test(R, 0, (N, 32, 64), 1; atype=Complex{Float64})

    # Regions:
    # test(R, 0, (N, ), 1; region=(N, ))
    # test(R, 0, (N, 64), 1; region=(N, 32:2:64))
    # test(R, 0, (N, 32, 64), 1; region=(N, 16, 32:2:64))
    test(R, 0, (N, ), 1; region=CartesianIndices((N, )))
    test(R, 0, (N, 64), 1; region=CartesianIndices((N, 32:2:64)))

    # Save/load:
    E = ones(N)
    # hts = plan_dht(R, E; save=true)
    # htl = plan_dht("plan_dht.jld2")
    # rm("plan_dht.jld2")
    hts = plan(R, E; save=true)
    htl = plan("dht.jld2")
    rm("dht.jld2")

    @test fieldnames(typeof(hts)) == fieldnames(typeof(htl))
    for v in fieldnames(typeof(hts))
        @test getfield(hts, v) == getfield(htl, v)
    end

    # AbstractFFTs API:
    E1 = ones(N)
    E2 = ones(N)
    # ht = plan_dht(R, E1)
    ht = plan(R, E1)

    dht!(E1, ht)
    ht * E2
    @test E2 == E1

    idht!(E1, ht)
    ht \ E2
    @test E2 == E1
end


@testset "CUDA" begin
    if has_cuda()
        CUDA.allowscalar(false)

        for p in [0, 1, 4]
            # 1D:
            test(R, p, (N, ), 1; atype=Float32, cuda=true)

            # 2D:
            test(R, p, (N, 64), 1; atype=Float32, cuda=true)
            # test(R, p, (64, N), 2; atype=Float32, cuda=true)

            # 3D:
            # test(R, p, (N, 32, 64), 1; atype=Float32, cuda=true)
            # test(R, p, (32, N, 64), 2; atype=Float32, cuda=true)
            # test(R, p, (32, 64, N), 3; atype=Float32, cuda=true)
        end

        # Different types:
        # 1D:
        test(R, 0, (N, ), 1; atype=Float64, cuda=true)
        test(R, 0, (N, ), 1; atype=Complex{Float64}, cuda=true)
        test(R, 0, (N, ), 1; atype=Complex{Float32}, cuda=true)

        # 2D:
        test(R, 0, (N, 64), 1; atype=Float64, cuda=true)
        test(R, 0, (N, 64), 1; atype=Complex{Float64}, cuda=true)
        test(R, 0, (N, 64), 1; atype=Complex{Float32}, cuda=true)

        # 3D:
        # test(R, 0, (N, 32, 64), 1; atype=Float64, cuda=true)
        # test(R, 0, (N, 32, 64), 1; atype=Complex{Float64}, cuda=true)
        # test(R, 0, (N, 32, 64), 1; atype=Complex{Float32}, cuda=true)

        # Regions:
        # test(R, 0, (N, ), 1; region=(N, ), cuda=true)
        # test(R, 0, (N, 64), 1; region=(N, 32:2:64), cuda=true)
        # test(R, 0, (N, 32, 64), 1; region=(N, 16, 32:2:64), cuda=true)
        test(R, 0, (N, ), 1; region=CartesianIndices((N, )), cuda=true)
        test(R, 0, (N, 64), 1; region=CartesianIndices((N, 32:2:64)), cuda=true)

        # Multiple repetitions to ensure no synchronization problems:
        # N1, N2 = 256, 500
        # E = zeros(Float32, (N1, N2))
        # for j=1:N2
        # for i=1:N1
        #     E[i, j] = exp(-i / N1) * exp(-j / N2)
        # end
        # end
        # E0 = copy(E)

        # E = CUDA.CuArray(E)
        # # ht = plan_dht(10, E)
        # ht = plan(10f0, E)

        # for i=1:100
        #     dht!(E, ht)
        #     idht!(E, ht)
        # end

        # @test isapprox(collect(E), E0)

        # Save/load:
        E = CUDA.ones(N)
        # hts = plan_dht(R, E; save=true)
        # htl = plan_dht("plan_dht.jld2")
        # rm("plan_dht.jld2")
        hts = plan(R, E; save=true)
        htl = plan("dht.jld2")
        rm("dht.jld2")

        @test fieldnames(typeof(hts)) == fieldnames(typeof(htl))
        for v in fieldnames(typeof(hts))
            @test getfield(hts, v) == getfield(htl, v)
        end

        # AbstractFFTs API:
        E1 = CUDA.ones(N)
        E2 = CUDA.ones(N)
        # ht = plan_dht(R, E1)
        ht = plan(R, E1)

        dht!(E1, ht)
        ht * E2
        @test E2 == E1

        idht!(E1, ht)
        ht \ E2
        @test E2 == E1
    end
    end
