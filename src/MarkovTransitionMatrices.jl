__precompile__()

module MarkovTransitionMatrices

using Distributions

export markov_transition, markovswitching_transition

myDist(μ::Real, σ::Real) = Normal(μ, σ)
myDist(μ::Vector, Σ::Matrix) = MvNormal(μ, Σ)
mypdf(dist::UnivariateDistribution, s2) = pdf(dist, s2...)
mypdf(dist::MultivariateDistribution, s2) = pdf(dist, [s2...])

"""
    markov_transition(μ::Function, Σ::Function, minp::AbstractFloat, statevectors::AbstractVector...)

Returns the sparse, **TRANSPOSED** Markov Transition Matrix (that is, `P[i,j] = Pr(i|j)`) where all
elements are `> minp`. The state-space is defined as the Cartesian Product of the `statevectors`.
The functions `μ` and `Σ` should take a tuple from `Base.product(statevectors...)` and return EITHER
  - the mean and **STANDARD DEVIATION** of `Distributions.Normal`, or
  - the mean and **VARIANCE MATRIX** of `Distributions.MvNormal`
"""
function markov_transition{T<:AbstractFloat}(μ::Function, Σ::Function, minp::AbstractFloat, statevectors::AbstractVector{T}...)

  0. <= minp < 1. || throw(DomainError())

  state_prod = Base.product(statevectors...)

  transposeP = zeros(T, length(state_prod), length(state_prod))

  for (j,s1) in enumerate(state_prod)
    dist = myDist(μ(s1), Σ(s1))
    for (i, s2) in enumerate(state_prod)
      transposeP[i,j] = mypdf(dist, s2)
    end
  end
  transposeP ./= sum(transposeP, 1)
  transposeP .= (transposeP .> minp) .* transposeP

  return sparse(transposeP)

end


"""
    markovswitching_transition(μ::Function, Σ::Function, π::Matrix{Float64}, minp::AbstractFloat, statevectors::AbstractVector...)

Returns a sparse, **TRANSPOSED** Markov Transition Matrix (that is, `P[i,j] = Pr(i|j)`) where all
elements are `> minp`. The state space is defined as the Cartesian Product of the
`statevectors` with `1:size(π,1)`. The functions `μ(::Int, ::Tuple)` and `Σ(::Int, ::Tuple)`
should take the index of a regime `1:size(π,1)` and a tuple from `Base.product(statevectors...)`
and return EITHER
  - the mean and **STANDARD DEVIATION** of `Distributions.Normal`, or
  - the mean and **VARIANCE MATRIX** of `Distributions.MvNormal`
The markov-switching matrix is NOT transposed and equals `π[i,j] = Pr(j|i)`

"""
function markovswitching_transition{T<:AbstractFloat}(μ::Function, Σ::Function, π::Matrix{Float64}, minp::AbstractFloat, statevectors::AbstractVector{T}...)

  k = size(π,1)
  k == size(π,2)   || throw(DimensionMismatch())
  all(sum(π,2) .== 1.0) || throw(error("each row of π must sum to 1"))


  n = prod(map(length, statevectors))
  regimes = Base.OneTo(k)
  transposeP = zeros(T, k*n, k*n)

  for r1 in regimes
    for r2 in regimes
      transposeP[(r2-1)*n+1:r2*n, (r1-1)*n+1:r1*n] .= π[r1, r2] .* markov_transition( (s) -> μ(r1, s), (s) -> Σ(r1, s), minp, statevectors... )
    end
  end

  return sparse(transposeP)
end


# module end
end
