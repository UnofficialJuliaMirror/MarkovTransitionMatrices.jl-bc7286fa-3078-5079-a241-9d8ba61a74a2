using MarkovTransitionMatrices

# ------------------ arbitrary process ---------------------

# state grids
grid_x1 = 0.0:1.0:10.0
grid_x2 = -1.5:0.5:15.0

# probabilities must all be greater than minp
minp = 1e-8

# Correlated random walk
μ(s) = [s...]
Σ(s) = [1.0 0.5; 0.5 1.0]
P = markov_transition(μ, Σ, minp, grid_x1, grid_x2)

# Markov-switching process

# regime transition matrix (NOT tranpsosed! - rows sum to 1.0)
πswitch = [.9 .1; .4 .6]

μswitch(r::Int, s) = r==1 ? [s...] : [s...] .+ ones(2)
Σswitch(r::Int, s) = r==1 ? eye(2) : [1.0 0.5; 0.5 1.0]
Pswitch = markovswitching_transition(μswitch, Σswitch, πswitch, 1e-8, grid_x1, grid_x2)


# ----------------------------- Farmer-Toda VAR method -------------------------------

# make price process
# with nσ_p = 5.0, discretized NG prices go from 1/3.19 to 3.49 times the min & max
σ2_p = 0x1.d1780728c185cp-9 # ≊  0.003551246  # OLD WAS: 0.01441732
μ_p  = 0x1.0640177c13d82p-6 # ≊ 0.01600649
α_p  = 0x1.fac3ea1ef1eadp-1 # ≊  0.98977596
nσ_p = 5.0

@show (σ2_p, μ_p, α_p,)

# OPTIONS: transition matrix
P_MIN_PROB = 1e-8     # min non-zero probability
STRIPE_RANGE = -50:50 # non-zero diagonals

NUM_P, STRIPE_RANGE = (75, -12:12,)

println("making VAR process")
p_process = VAR_process(μ_p, α_p, σ2_p)
p_transitions = VAR_states_transition(p_process, NUM_P, 2, nσ_p)
Π_p = p_transitions.P'
Π_p = sparsify_transition_matrix(p_transitions.P, P_MIN_PROB, STRIPE_RANGE)'
P_GRID = vec(p_transitions.X)
