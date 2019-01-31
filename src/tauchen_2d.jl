export tauchen_2d!, tauchen_1d!, bvn_discreteApprox!, tauchen_2d, tauchen_1d

function bvn_upperlower_cdf(xlim::NTuple{2}, ylim::NTuple{2}, r::Float64)
    xl,xu = xlim
    yl,yu = ylim
    xl < xu && yl < yu || throw(DomainError())
    return bvncdf(xu,yu,r) - bvncdf(xl,yu,r) - bvncdf(xu,yl,r) + bvncdf(xl,yl,r)
end


function plus_minus_dx(x::Number, xspace::AbstractRange, scaling::Number=1)
  dx = step(xspace)/2
  x == first(xspace) && return (-Inf, dx,).*scaling
  x == last(xspace)  && return (-dx, Inf,).*scaling
  return (-dx, dx).*scaling
end

x_pm_Δ(x::Number, xspace::AbstractVector) = x .+ plus_minus_dx(x,xspace)

function tauchen_1d!(P::AbstractMatrix, S::AbstractVector, μ::Function, σ2::Number)
    σ2 > 0 || throw(DomainError())
    all(length(S) .== size(P)) || throw(DimensionMismatch())

    sigma = sqrt(σ2)
    invsigma = inv(sigma)

    for (j,s1) in enumerate(S)
      pm_dz = plus_minus_dx(s1, S, invsigma)
      for (i,s0) in enumerate(S)
        zm, zp = ((s1 - μ(s0))/sigma) .+ pm_dz
        P[i,j] = normcdf(zp) - normcdf(zm)
      end
    end
end

function tauchen_1d(S::AbstractVector{T}, μ::Function, σ2::Number) where {T<:Real}
    n = length(S)
    P = Matrix{T}(undef,n,n)
    tauchen_1d!(P,S,μ,σ2)
    return P
end

function tauchen_2d(S::Base.Iterators.ProductIterator, μ::Function, Σ::AbstractMatrix{T}) where {T<:Real}
  n = length(S)
  P = Matrix{T}(undef,n,n)
  tauchen_2d!(P,S,μ,Σ)
  return P
end

function tauchen_2d!(P::AbstractMatrix, S::Base.Iterators.ProductIterator, μ::Function, Σ::AbstractMatrix)
    ndims(S) == 2
    (2,2,) == size(Σ) || throw(DimensionMismatch("Σ must be 2x2"))
    issymmetric(Σ) || throw(error("Σ must be symmetric"))
    all(length(S) .== size(P)) || throw(DimensionMismatch())

    sigmas = sqrt.(diag(Σ))
    invsigmas = inv.(sigmas)
    ρ = Σ[1,2] / prod(sigmas)

    for (j,s1) in enumerate(S)
      pm_dz = plus_minus_dx.(s1, S.iterators, invsigmas)
      for (i,s0) in enumerate(S)
        z = (s1 .- μ(s0))./sigmas
        P[i,j] = bvn_upperlower_cdf(z[1] .+ pm_dz[1], z[2] .+ pm_dz[2], ρ)
      end
    end
end


function bvn_discreteApprox!(P::AbstractMatrix{T}, S::Base.Iterators.ProductIterator, μ::Function, Σ::AbstractMatrix; κ=1e-8) where {T<:Real}

  nS = length(S)
  ndims(S) == 2 == size(Σ,1) == size(Σ,2) || throw(DimensionMismatch())
  (nS,nS) == size(P)  || throw(DimensionMismatch())
  issymmetric(Σ) || throw(error("Σ must be symmetric"))

  # Initialize elements that will be returned
  maxMoments = 5
  Λ          = zeros(T, nS, maxMoments)
  JN         = zeros(T, nS)
  approxErr  = zeros(T, nS, maxMoments)
  numMoments = zeros(Int, nS)

  # preallocate these, which will be updated each iteration
  ΔT  = Array{T}(undef,nS,maxMoments)
  q   = Array{T}(undef,nS)
  q1  = similar(q)
  tmp = similar(q)
  @views z = ΔT[:,1:2]

  tauchen_2d!(P, S, μ, Σ)

  sigmas = sqrt.(diag(Σ))
  invsigmas = inv.(sigmas)
  ρ = Σ[1,2] / prod(sigmas)


  # state s0
  for (i,s0) in enumerate(S)
    for (j,s1) in enumerate(S)
      z[j,:] .= (s1 .- μ(s0)) ./ sigmas
    end
    ΔT[:,3:4] .= z.^2 .- 1.0
    ΔT[:,5] .= z[:,1].*z[:,2] .- ρ

    @views q .= P[i,:]
    q1 .= max.(q,κ)

    # attempt to match mean, variance, correlation
    updated = false
    for l in 5:-1:4
      @views J = discreteApprox!(P[i,:], Λ[i,1:l], approxErr[i,1:l], tmp, q1, ΔT[:,1:l])
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
