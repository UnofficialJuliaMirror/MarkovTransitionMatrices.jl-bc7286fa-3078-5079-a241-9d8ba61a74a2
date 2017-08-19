# __precompile__()

module MarkovTransitionMatrices

using Distributions
using Optim

export markov_transition, markovswitching_transition, markov_transition_moment_matching

myDist(μ::Real, σ::Real) = Normal(μ, σ)
myDist(μ::Vector, Σ::Matrix) = MvNormal(μ, Σ)
mypdf(dist::UnivariateDistribution, s2) = pdf(dist, s2...)
mypdf(dist::MultivariateDistribution, s2) = pdf(dist, [s2...])

include("simple_no_matching.jl")
include("moment_matching.jl")

# module end
end
