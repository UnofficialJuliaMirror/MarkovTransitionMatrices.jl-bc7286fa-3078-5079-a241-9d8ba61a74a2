using MarkovTransitionMatrices
using StatsFuns
using Base.Iterators
using Plots
gr()


ρ = 0.5
num_moments_to_match = 5
num_y_nodes = 11
num_yy_nodes = num_y_nodes^2
ylim = 3.0

yspace = range(-abs(ylim); stop=abs(ylim), length=num_y_nodes)
yyspace = product(yspace,yspace)
dT = zeros(num_yy_nodes, num_moments_to_match)
P = Matrix{Float64}(undef, num_yy_nodes, num_yy_nodes)

function bvn_upperlower_cdf(xlim::NTuple{2}, ylim::NTuple{2}, r::Float64)
    xl,xu = xlim
    yl,yu = ylim
    xl < xu && yl < yu || throw(DomainError())
    return bvncdf(xu,yu,r) - bvncdf(xl,yu,r) - bvncdf(xu,yl,r) + bvncdf(xl,yl,r)
end

function x_pm_Δ(x::Number,Δ::Number)
  return (x-Δ, x+Δ,)
end

function x_pm_Δ(x::Number, Δ::Number, xspace::AbstractVector)
  if x == first(xspace)
    return (-Inf, x+Δ,)
  elseif x == last(xspace)
    return (x-Δ, Inf)
  else
    return x_pm_Δ(x,Δ)
  end
end


function tauchen_2d!(P::AbstractMatrix, μ::Function, Σ::AbstractMatrix, zprod::Base.Iterators.ProductIterator)
    ndims(zprod) == 2
    (2,2,) == size(Σ) || throw(DimensionMismatch("Σ must be 2x2"))
    issymmetric(Σ) || throw(error("Σ must be symmetric"))

    all(length(zprod) .== size(P)) || throw(DimensionMismatch())
    Δs = step.(zprod.iterators)./2

    σ1,σ2 = sqrt.(diag(Σ))
    ρ = Σ[1,2] / (σ1*σ2)

    for (j,zj) in enumerate(zprod)
      for (i,zi) in enumerate(zprod)
        zjpm1, zjpm2 = x_pm_Δ.(zj, Δs, zprod.iterators)
        Ezj = μ(zi)
        P[i,j] = bvn_upperlower_cdf((zjpm1 .- Ezj[1])./σ1, (zjpm2 .- Ezj[2])./σ2, ρ)
      end
    end
end

PP = zeros(num_yy_nodes, num_yy_nodes)
tauchen_2d!(PP, (x) -> x, [1.0 0.5; 0.5 1.0], yyspace)
heatmap(reshape(PP[73,:],11,11))
@test all(sum(PP, dims=2) .≈ 1.0)



function ΔTmat_bvn!(dT::Matrix, sspace::Base.Iterators.ProductIterator, s0, Es1, Sigma::AbstractMatrix)
  size(dT) == length(sspace), 5) || throw(error(DimensionMismatch()))
  length(s0) == length(Es1) == 2  || throw(error(DimensionMismatch()))
  size(Sigma) == (2,2,)  || throw(error(DimensionMismatch()))

  for j in 1:2
    # compute deviation from mean
    for (i,s) in enumerate(sspace)
      dT[i,j] .= s[j] .- Es1[j]
    end
    # compute variance terms
    @views dT[:,2+j] .= dT[:,j].^2 .- Sigma[j,j]
  end
  # compute covariance
  @views dT[:,5] .= dT[:,1].*dT[:,2] .- Sigma[1,2]
end


function plus_minus_dx(x::Number, xspace::AbstractRange)
  dx = step(xspace)/2
  x == first(xspace) && return (-Inf, dx,)
  x == last(xspace)  && return (-dx, Inf)
  return (-dx, dx)
end

function discreteApprox!(P::AbstractMatrix, S::Base.Iterators.ProductIterator, μ::Function, Σ::AbstractMatrix) where {T<:Real}

  nS = length(S)
  ndims(S) == 2 == size(Σ,1) == size(Σ,2) || throw(DimensionMismatch())
  (nS,nS) == size(P)  || throw(DimensionMismatch())
  maxMoments = 5

  # Initialize elements that will be returned
  Λ          = zeros(T, nS, maxMoments)
  JN         = zeros(T, nS)
  approxErr  = zeros(T, nS, maxMoments)
  numMoments = zeros(Int, nS)

  # preallocate these, which will be updated each iteration
  ΔT  = Array{T}(undef,nS, maxMoments)
  z   = Array{T}(undef,nS, 2)
  q   = Array{T}(undef,nS)
  tmp = Array{T}(undef,nS)

  sigmas = sqrt.(diag(Σ))
  ρ = Σ[1,2] / prod(sigmas)

  # state s0
  for (i,s0) in enumerate(S)

    # z ≡ (s₁ - E[s₁|s₀])/σ
    for (j,s1) in enumerate(S)
      z = (s1 .- μ(s0)) ./ sigmas
      pm_ds = plus_minus_dx.(s1, S.iterators) ./ sigmas
      qj = bvn_upperlower_cdf( z[1] .+ pm_ds[1], z[2] .+ pm_ds[2], ρ)
      q[j] = max(qj, κ)
    end

    z ./= scale_factor
    ΔTmat!(ΔT, z, [0.0, 0.0, 1.0, 1.0, ρ])
    updated = false

    for l in 5:-1:2
      @views J = discreteApprox!(P[i,:], Λ[i,1:l], approxErr[i,1:l], tmp, q, ΔT[:,1:l])
      if isfinite(J)
        JN[i], numMoments[i] = (J, l)
        updated = true
        break
      end
    end
    if !updated
      sumq = sum(q)
      P[i,:] .= q ./ sumq
    end

  end

  return JN, Λ, numMoments, approxErr
end







# make deviations
zrandwalk(x::Real, st::Real, σ::Real) = (x - st) / σ

# create log σ discretization
Pσ, JN, Λ, L_p1, approxErr = discreteNormalApprox(logσ_space, logσ_space, (x::Real,st::Real) -> zrandwalk(x,st,sdlogσ), 7)
all(sum(Pσ,dims=2) .≈ 1.0) || throw(error("each row of π must sum to 1"))

# match moments for logp
p_from_sigma(logσ::Real) = Dict( [:P, :JN, :Λ, :L, :approxErr,] .=> discreteNormalApprox(logp_space, logp_space, (x::Real,st::Real) -> zrandwalk(x, st, exp(logσ)), 11) )
c_from_sigma(logσ::Real) = Dict( [:P, :JN, :Λ, :L, :approxErr,] .=> discreteNormalApprox(logcspace, logcspace, (x::Real,st::Real) -> zrandwalk(x, st, exp(logσ)*cσ_to_pσ), 11) )

ps_from_sigma = Dict(logσ_space .=> [p_from_sigma(logσ) for logσ in logσ_space])
cs_from_sigma = Dict(logσ_space .=> [c_from_sigma(logσ) for logσ in logσ_space])

# Fill giant transition matrix
P = Matrix{eltype(logp_space)}(undef, nlogp*nlogc*nlogσ, nlogp*nlogc*nlogσ)
for (j, logσ) in enumerate(logσ_space)
  for i in 1:nlogσ
    P[blockrange(i,j,nlogp*nlogc)...] .= Pσ[i, j] .* kron( cs_from_sigma[logσ][:P],  ps_from_sigma[logσ][:P]) # ps_from_sigma[logσ][:P]
  end
end

# sparsify transigion
Πp = MarkovTransitionMatrices.sparsify!(P, minp)
