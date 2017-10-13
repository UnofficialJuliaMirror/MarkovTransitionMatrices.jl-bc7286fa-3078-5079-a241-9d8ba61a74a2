using MarkovTransitionMatrices
using Distributions

MarkovTransitionMatrices.Tbar(5,1.0)


σp = [0.05028345, 0.07192626]
σc = [0.008112649, 0.036694635]

Πk = [0.93517852 0.07835965 ;
      0.06482148 0.92164035 ]

extrema_p = [0.8776572, 2.485073]
extrema_c = [0.2437302, 1.529529]

pspace = linspace(extrema_p[1]-log(2.), extrema_p[2] + log(2.), 11)
cspace = linspace(extrema_c[1]-log(2.), extrema_c[2] + log(2.), 11)

function expΔTx!(x::Vector, ΔT::AbstractMatrix, tmpvec::Vector)
  A_mul_B!(tmpvec, ΔT, x)
  tmpvec .= exp.(tmpvec)
end

# objective
function f2!(tmpvec::Vector, x::Vector, q::Vector, ΔT::AbstractMatrix)
  expΔTx!(tmpvec, ΔT, x)
  return dot(q, tmpvec)
end

# gradient
function g2!(grad::Vector, tmpvec::Vector, x::Vector, q::Vector, ΔT::AbstractMatrix)
  expΔTx!(tmpvec, ΔT, x)
  tmpvec .*= q
  grad .= sum(tmpvec .* ΔT)
end

function fg2!(grad::Vector, tmpvec::Vector, x::Vector, q::Vector, ΔT::AbstractMatrix)
  expΔTx!(tmpvec, ΔT, x)
  tmpvec .*= q
  grad .= sum(tmpvec .* ΔT)
  return sum(tmpvec)
end

function transition!(P::AbstractMatrix, y::AbstractVector{T}, S::Union{AbstractVector, Base.AbstractProdIterator}, sdev::Function, maxMoments::Int=2, κ::Real=1e-8) where {T<:Real}

  num_states = length(S)
  n = length(y)
  maxMoments >= n   || throw(error("Must use $n-1 moments or vewer"))
  (nS,n) == size(P) || throw(DimensionMismatch())

  # Initialize elements that will be returned
  Λ          = zeros(T  , nS, maxMoments)
  JN         = zeros(T  , nS)
  approxErr  = zeros(T  , nS, maxMoments)
  numMoments = zeros(Int, nS)

  # preallocate these, which will be updated each iteration
  ΔT   = Array(T,n,maxMoments)
  dev  = Array(T,n)
  q    = Array(T,n)
  tmpvec = Array(T,n)

  δ = y[end]  # a scaling factor
  Tbar = [m for m in NormCentralMoment(L, 1./δ)]

  for i,s in enumerate(S)
      dev .= sdev(y, s)
      q .= max.(normpdf(dev), κ)
      dev ./= δ

      for l in maxMoments:-1:1
        for j = 1:l
          ΔT[:,j] .= dev.^j .- Tbar[j]
        end

        # closures
        ΔTvw = @view(ΔT[:,1:l])
        odf = OnceDifferentiable(
          (x::Vector)               -> f2!(       tmpvec, x, q, ΔT),
          (grad::Vector, x::Vector) -> g2!( grad, tmpvec, x, q, ΔT),
          (grad::Vector, x::Vector) -> fg2!(grad, tmpvec, x, q, ΔT)
        )

        # optimize to match moments
        try
          res = Optim.optimize(odf, ones(T,l))
          λ = Optim.minimizer(res)
          J_candidate = Optim.minimum(res)
          grad = zeros(T,l)
          g!(λ,grad)

          # if we like the results, update and break
          if ( norm( grad ./ J_candidate ) < 1e-5 ) & all(isfinite.(grad)) & all(isfinite.(λ)) & (J_candidate > 0.0)
            JN[i] = J_candidate
            Λ[i,1:l] .= λ
            for j in 1:n
              expΔTx!(tmpvec, ΔTvw, λ)
              P[i,:] .= q .* tmpvec ./ J_candidate
            end
            approxErr[i,1:l] .= grad ./ J_candidate
            numMoments[i] = l
            break
          end # if statment
        catch
        end

      end   # loop over moment number of conditions (l=maxMoments:1)
    end     # loop over number of states (m = 1:M)
  end       # loop over state space (i=1:J)

  return P, JN, Λ, numMoments, approxErr
end




#
