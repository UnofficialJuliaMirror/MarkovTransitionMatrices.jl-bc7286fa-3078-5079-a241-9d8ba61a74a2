# using MarkovTransitionMatrices

# state grids
grid_x1 = 0.0:1.0:10.0
grid_x2 = -1.5:0.5:15.0

# probabilities must all be greater than minp
minp = 1e-8

# Correlated random walk
μ(s) = [s...]
Σ(s) = [1.0 0.5; 0.5 1.0]
P = markov_transition(μ, Σ, minp, grid_x1, grid_x2)

full(markov_transition((s)-> 0.0, (s...) -> 1.0, minp, -3.0:0.3:3.0))[1,:]

MarkovTransitionMatrices.

# Markov-switching process

# regime transition matrix (NOT tranpsosed! - rows sum to 1.0)
πswitch = [.9 .1; .4 .6]

μswitch(r::Int, s) = r==1 ? [s...] : [s...] .+ ones(2)
Σswitch(r::Int, s) = r==1 ? eye(2) : [1.0 0.5; 0.5 1.0]
Pswitch = markovswitching_transition(μswitch, Σswitch, πswitch, 1e-8, grid_x1, grid_x2)
