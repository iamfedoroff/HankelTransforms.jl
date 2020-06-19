R = 3f0
N = 256

v = HankelTransforms.htfreq(R, N)
r = HankelTransforms.htcoord(R, N)
f = @. mysinc(r)
f = CUDA.CuArray{typeof(R)}(f)

plan = HankelTransforms.plan(R, f)

f1 = copy(f)
f2 = copy(f)

HankelTransforms.dht!(f1, plan)
plan * f2
@test f1 == f2

HankelTransforms.idht!(f1, plan)
plan \ f2
@test f1 == f2
