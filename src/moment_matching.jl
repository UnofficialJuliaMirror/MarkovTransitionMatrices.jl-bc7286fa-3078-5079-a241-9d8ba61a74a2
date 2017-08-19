
function lower_triangular!(x::AbstractMatrix{T}, ltvec::AbstractVector{T}) where {T}
  n,m = size(x)
  n == m || throw(DimensionMismatch("must be square"))
  length(ltvec) == n*(n+1)/2
  k = 0
  for j = 1:n
    for i = j:n
      k += 1
      ltvec[k] = x[i,j]
    end
  end
  return ltvec
end

function lower_triangular(x::AbstractMatrix{T}) where {T}
  n = size(x, 1)
  ltvec = Vector{T}(Int(n*(n+1)/2))
  get_lower_triangular(x, ltvec)
end

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

function f(λ::Vector{T}, q0::Vector{T}, ΔT::Matrix{T}) where {T<:AbstractFloat}
  (L,J) = size(ΔT)
  J == length(q0) || throw(DimensionMismatch())
  L == length(λ) || throw(DimensionMismatch())
  val = zero(T)
  for j = 1:J
    val += q0[j] * exp(dot(λ, ΔT[:,j]))
  end
  return val
end

function g!(λ::Vector{T}, grad::Vector{T}, q0::Vector{T}, ΔT::Matrix{T}) where {T<:AbstractFloat}
  (L,J) = size(ΔT)
  J == length(q0) || throw(DimensionMismatch())
  L == length(λ) == length(grad) || throw(DimensionMismatch())
  grad .= 0.0
  for j = 1:J
    x = q0[j] * exp(dot(λ, ΔT[:,j]))
    grad .+= x .* ΔT[:,j]
  end
end


function h!(λ::Vector{T}, hess::Matrix{T}, q0::Vector{T}, ΔT::Matrix{T}) where {T<:AbstractFloat}
  (L,J) = size(ΔT)
  J == length(q0) || throw(DimensionMismatch())
  L == length(λ) || throw(DimensionMismatch())
  (L,L,) == size(hess) || throw(DimensionMismatch())
  hess .= zero(T)
  for j = 1:J
    x = q0[j] * exp(dot(λ, ΔT[:,j]))
    Base.LinAlg.BLAS.gemm!('N', 'T', x, ΔT[:,j], ΔT[:,j], 1.0, hess)
  end
end

# --------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


# Multivariate version
function momentdiff!(sprod::Base.Iterators.AbstractProdIterator, theory_mean::Vector{T}, theory_var::Matrix{T}, ΔT::Matrix) where {T<:AbstractFloat}
  J = length(sprod)
  nμ = length(theory_mean)
  (nμ, nμ) == size(theory_var) || throw(error(DimensionMismatch()))
  (nμ + nμ*(nμ+1)/2, J) == size(ΔT) || throw(error(DimensionMismatch()))

  dev = zeros(T, ndims(sprod))
  outerprod = zeros(T, ndims(sprod), ndims(sprod))

  for (j,s) in enumerate(sprod)
    dev .= [s...] .- theory_mean
    outerprod .= (dev * dev') .- theory_var
    ΔT[1:nμ, j] .= dev
    lower_triangular!(outerprod, @view(ΔT[nμ+1:end, j]))
  end
  return nothing
end


# Univariate version
function momentdiff!(sprod::Base.Iterators.AbstractProdIterator, theory_mean::T, theory_sd::T, ΔT::Matrix) where {T<:AbstractFloat}
  J = length(sprod)
  ndims(sprod) == 1 || throw(DimensionMismatch())
  (2, J) == size(ΔT) || throw(DimensionMismatch())
  theory_var = theory_sd^2

  for (j,s) in enumerate(sprod)
    dev = s[1] - theory_mean
    outerprod = dev^2 - theory_var
    ΔT[:,j] .= [dev, outerprod]
  end
  return nothing
end


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------



function markov_transition_moment_matching(μ::Function, Σ::Function, minp::AbstractFloat, statevectors::AbstractVector{T}...) where {T<:AbstractFloat}

  0. <= minp < 1. || throw(DomainError())

  state_prod = Base.product(statevectors...)

  # dimensions
  nd = ndims(state_prod)     # num basis vectors in state space
  J = length(state_prod)     # size of state space
  L = Int(nd + nd*(nd+1)/2)  # moments to match (mean + var)

  # temp variables
  q0 = zeros(T, J)
  ΔT = zeros(T, L, J)
  grad = zeros(T, L)

  # output variables
  P = zeros(T, length(state_prod), length(state_prod))
  approxErr = fill(typemax(T), L, J)
  moments_matched = zeros(Int, J)

  # Need to parallelize this?
  for (i, s1) in enumerate(state_prod)
    mean0 = μ(s1)
    var0 =  Σ(s1)

    dist = myDist(mean0, var0)
    q0 .= 0.0

    # fill in initial approximation
    for (j, s2) in enumerate(state_prod)
      q0[j] = mypdf(dist, s2)
    end

    momentdiff!(state_prod, mean0, var0, ΔT)

    f_cl( λ::Vector{T})                  where {T} = f( λ,       q0, ΔT)
    g_cl!(grad::Vector{T}, λ::Vector{T}) where {T} = g!(λ, grad, q0, ΔT)
    h_cl!(hess::Matrix{T}, λ::Vector{T}) where {T} = h!(λ, hess, q0, ΔT)

    res = Optim.optimize(f_cl, g_cl!, h_cl!, ones(T,L))
    λ = Optim.minimizer(res)
    J_candidate = Optim.minimum(res)
    g_cl!(grad, λ)
    approxErr[:, i] = grad / J_candidate

    # if we like the results, update and break
    if ( norm( grad ./ J_candidate ) < 1e-4 ) & all(isfinite.(grad)) & all(isfinite.(λ)) & (J_candidate > 0.0)
      for k in 1:length(q0)
        P[i,k] = q0[k] * exp( dot(λ, ΔT[:,k]) ) / J_candidate
      end
      moments_matched[i] = L
    else
      P[i,:] .= q0
    end
  end

  P ./= sum(P, 2)
  P .= (P .> minp) .* P
  P ./= sum(P, 2)

  return sparse(P), moments_matched, approxErr
end
