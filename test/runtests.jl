using MarkovTransitionMatrices
using Base.Test

using Distributions

isapprox_oneval(x::AbstractArray) = all( x .≈ x[1] )

# setup for test
testvec = -3.0:0.5:0.0
ss = collect(Base.product(testvec, testvec))
n = length(testvec)
nn = length(ss)
mvd = MvNormal(zeros(2), eye(2))

# test the 1-d version
fullp1 = full(markov_transition((s) -> 0., (s) -> 1., 1e-8, testvec))
ratio1 = fullp1 ./ pdf(Normal(0,1), testvec')
@test isapprox_oneval(ratio1)

# test the 2-d version
fullp2 = full(markov_transition((s) -> zeros(2), (s) -> eye(2), 1e-8, testvec, testvec))
ratio2 = reshape(fullp2[1,:], n, n) ./  [pdf(mvd, [s...]) for s in ss]
@test isapprox_oneval(ratio2)

# test the markov-switching version
μswitch(r::Real,s::NTuple{N,T}) where {N,T<:Real} = r==1 ? zeros(2) : [s...]
Σswitch(r::Real,s::NTuple{N,T}) where {N,T<:Real} = r==1 ? eye(2)   : eps(1.)*eye(2)
πswitch = eye(Float64, 2)
fullp3 = full(markovswitching_transition(μswitch, Σswitch, πswitch, 0.0, testvec, testvec))

@test all(fullp3[1:nn, 1:nn] .== fullp2)
@test all(fullp3[nn+1:2*nn, 1:nn] .== 0.0)
@test all(fullp3[1:nn, nn+1:2*nn] .== 0.0)
@test all(fullp3[nn+1:2*nn, nn+1:2*nn] .== eye(nn))
