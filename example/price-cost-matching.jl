using SparseArrays
using MarkovTransitionMatrices
using StatsFuns
using Base.Iterators

# sparse versions
minp = 1e-4
extrema_logp = [0.8510954,  2.405270]
extrema_logc = [5.0805023,  5.318393]
extrema_logσ = [-2.9558226, -1.542508]
sdlogσ = 0.18667531
cσ_to_pσ = 0.02777901 / 0.08488665

nlogp = 25
nlogc = 15
nlogσ = 9

# range for i,j block in matrix where block is n x n
blockrange(i,j,n) = ((i-1)*n+1):(i*n),  ((j-1)*n+1):(j*n)

# make range for coefs
logσ_space = range(extrema_logσ[1] - 2*sdlogσ, stop=extrema_logσ[2] + 2*sdlogσ, length=nlogσ)
logp_space = range(extrema_logp[1] - log(2.0), stop=extrema_logp[2] + log(2.0), length=nlogp)
logcspace = range(extrema_logc[1] - log(2.0), stop=extrema_logc[2] + log(2.0), length=nlogc)


sspace = product(logp_space, logcspace,)
dT = zeros(length(sspace), 5)

meanou(x,k0,k1) = x + k0 + k1*exp(x)
meanlogp(x) = meanou(x, -0.017942728,  0.003373849)
meanlogc(x) = meanou(x, -0.0236377335, 0.0001336044)

s0 = logp_space[13], logcspace[8]
Es1 = meanlogp(s0[1]), meanlogc(s0[2])

sdp = exp(logv)
sdc = cσ_to_pσ*vp

vars = sdp^2, sdc^2
covpc = rho*sdc*sdp

function ΔTmat_bvn!(dT::Matrix, sspace::Base.Iterators.ProductIterator, s0, Es1, Sigma::AbstractMatrix)
  size(dT) == length(sspace, 5) || throw(error(DimensionMismatch()))
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


scale_factor = maximum(abs.(y))
scaled_moments = [m for m in NormCentralMoment(maxMoments, 1.0/scale_factor)]
discreteApprox!(P, y, S, zval, normpdf, scaled_moments, scale_factor, maxMoments, κ)

function discreteApprox!(P::AbstractMatrix, y::AbstractVector{T}, S::Union{AbstractVector, Base.Iterators.ProductIterator}, zval::Function, pdffun::Function, scaled_moments::Vector, scale_factor::Real, maxMoments::Integer, κ::Real) where {T<:Real}

  nS = length(S)
  n = length(y)
  0 < maxMoments < n || throw(error("Must use 1 to $n-1 moments or fewer"))
  (nS,n) == size(P)  || throw(DimensionMismatch())

  # Initialize elements that will be returned
  Λ          = zeros(T, nS, maxMoments)
  JN         = zeros(T, nS)
  approxErr  = zeros(T, nS, maxMoments)
  numMoments = zeros(Int, nS)

  # preallocate these, which will be updated each iteration
  ΔT  = Array{T}(undef,n, maxMoments)
  z   = Array{T}(undef,n)
  q   = Array{T}(undef,n)
  tmp = Array{T}(undef,n)

  # state s0
  for (i,s0) in enumerate(S)

    z .= zval.(y, s0)
    q .= max.(pdffun.(z), κ)
    z ./= scale_factor
    ΔTmat!(ΔT, z, scaled_moments)
    updated = false

    for l in maxMoments:-1:2
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
