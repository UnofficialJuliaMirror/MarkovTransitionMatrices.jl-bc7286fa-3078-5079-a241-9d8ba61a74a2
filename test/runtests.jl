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

# ---------------------------------------------------------------

s = -3.0:0.25:3.0
ss = -3.0:0.25:3.0, -3.0:0.25:3.0
# simulate random walks w/ moment matching
P_match_1, mom, er = markov_transition_moment_matching((s) -> 0.0   , (s) -> 1.0   , 1e-8, s)
P_match_2, mom, er = markov_transition_moment_matching((s) -> [s...], (s) -> eye(2), 1e-8, ss...)

# share of states that get moment-matching
share = sum(mom .> 0) / length(mom) * 100
@show "$share percent of states get moment-matching"
@test share .> 0.5

# without matching
P_nomatch_2 = markov_transition((s) -> [s...], (s) -> eye(2), 1e-8, ss...)
number_diff = sum(P_match_2 .!= P_nomatch_2)
sd_diff = sum((P_match_2 .- P_nomatch_2).^2)/prod(size(P_match_2))
abs_diff = sum(abs.(P_match_2 .- P_nomatch_2))/prod(size(P_match_2))
sharediff = number_diff / prod(size(P_match_2))
nm = norm(vec(P_match_2 .- P_nomatch_2), Inf)
@show "max difference between matching & no-matching is $nm"
@show "$sharediff percent of probabilities change"

# using Plots
# gr()
#
# heatmap(s, s, reshape(P_match_2[10,:]  , length.(ss)...))
# heatmap(s, s, reshape(P_nomatch_2[10,:], length.(ss)...))
# heatmap(s, s, reshape(P_match_2[20,:] .- P_nomatch_2[20,:], length.(ss)...))
# heatmap(s, s, reshape(mom.>0, length.(ss)...))
# histogram(vec(clamp.(P_match_2 .- P_nomatch_2, -0.05, 0.05)) )




#
