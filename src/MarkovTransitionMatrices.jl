# __precompile__()

"Functions to discretize a Markov transition matrix."
module MarkovTransitionMatrices

using Distributions
using NLopt
using Distributions
using Optim
using ProgressMeter
using StatsBase

using Base.Iterators

include("general_markov_transition.jl")
include("Farmer_Toda.jl")
include("striped_matrices.jl")

# module end
end
