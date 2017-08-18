export
  VAR_process,
  VAR_states_transition,
  sparsify_transition_matrix

# -------------------- types --------------------------

"Includes the states & transition for discretized VAR process"
struct VAR_states_transition{T<:AbstractFloat,A<:AbstractArray{T}, M<:AbstractMatrix{T}}
  X::A
  P::M
end


"VAR process object that we will discretize"
struct VAR_process{T<:AbstractFloat}
  b::Array{T}
  B::Array{T,2}
  Ψ::Array{T,2}
  M::Int
  A::Array{T,2}
  C::Array{T,2}
  Σ::Array{T,2}
  μ::Array{T,2}
end

# -------------------- cosntructors --------------------------

"Wrapper constructs markov Transition matrix & discrete states from VAR_process"
function VAR_states_transition(vpp::VAR_process, states_per_dim::Int=4, num_moments::Int=2, nσ::Real = 0.)
  yy = yspace(vpp, states_per_dim, nσ)
  SS = nprod(yy, vpp.M)
  DD = makeD(SS)

  P, JN, Λ, nM, er = makeTransitionMatrix(yy, SS, DD, vpp, num_moments)
  transitions = probabilityProd(P)
  countmap(nM)

  states = broadcast(+, vpp.C*DD, vpp.μ)

  return VAR_states_transition(states, transitions)
end






"Constructor for univariate AR process"
function VAR_process(b::T, B::T, Ψ::T) where {T<:AbstractFloat}
  o2 = ones(T, 1, 1)
  o1 = ones(T, 1)

  M = 1
  C = sqrt(Ψ)   * o2;
  A = B         * o2;
  μ = b/(1-B)   * o2;
  Σ = 1/(1-B^2) * o2;

  return VAR_process(b*o1, B*o2, Ψ*o2, M, A, C, Σ, μ)
end


"Constructor for VAR process"
function VAR_process(b::Array{T,1}, B::Array{T,2}, Ψ::Array{T,2}) where {T<:AbstractFloat}
  size(B) == size(Ψ)     || error("B and Ψ must be the same size")
  length(b) == size(B,1) || error("b must be same length as B's width")

  M = size(B,1)

  b = reshape(b,M,1)
  tildeC = chol(Ψ)'
  tildeA = tildeC \ B * tildeC
  tildeΣ = reshape( (eye(M^2) - kron(tildeA,tildeA)) \ reshape(eye(M), M^2, 1), M, M)

  # Objective is to make unconditional varances of y_k's equal
  function f(Vvec::Array{T,1}, grad::Array{T,1})
    V = reshape(Vvec,M,M)
    return norm( broadcast(-, diag(V'*tildeΣ*V), trace(tildeΣ)/M) )
  end

  # Constraint is that U is orthognoal (U'U = I)
  function c(res::Array{T,1}, Vvec::Array{T,1}, grad::Array{T,2})
    V = reshape(Vvec,M,M)
    res[:] = vec(V'*V - eye(M))
  end

  # find best U matrix such that y has same unconditional variance
  opt = Opt(:LN_COBYLA, M^2)
  min_objective!(opt, f)
  equality_constraint!(opt, c, 1e-12*ones(M^2))
  xtol_rel!(opt, 1e-12)
  (fObjOpt, xOpt, flag) = NLopt.optimize(opt, vec(eye(M)))

  # make transformation matrices
  U = reshape(xOpt,M,M)
  C = tildeC*U
  A = U'*tildeA*U
  μ = (eye(M) - B)\b
  Σ = U'*tildeΣ*U

  return VAR_process(b,B,Ψ,M,A,C,Σ,μ)
end



# -------------------- helper functions --------------------------


"Linear space for y given a VAR process and length Nm"
function yspace(vp::VAR_process{T}, Nm::Int=5, nσ::Real=0.0) where {T}
  σmin = sqrt(minimum(eigvals(vp.Σ)))
  if nσ <= zero(T)
    nσ = sqrt(convert(T, Nm) - one(T))
  end
  return linspace(-σmin*nσ, σmin*nσ, Nm)
end


"Make Cartesian Product given state space"
function nprod(x::AbstractVector{T}, n::Int) where {T<:Real}
  rep_x = ntuple((i) -> x, n)
  Base.product( rep_x... )
end

"Make state space matrix from vector"
makeD(S::AbstractVector) = reshape(collect(S), (1, length(S)) )


"Make state space matrix from product iterator"
function makeD(S::Base.Iterators.AbstractProdIterator)
  T = Float64
  D = zeros(T, length(size(S)), length(S) )
  for (j,s) in enumerate(S)
    D[:,j] = collect(s)
  end
  return D
end


"Moments of scaled normal distribution"
function Tbar(L::Int, δ_m::AbstractFloat)
   out = zeros(eltype(δ_m), L)
   nz = filter(iseven, 1:L)
   out[nz] = [2^(-k/2) * factorial(k) / factorial(k/2) * (1/δ_m)^k for k in nz]
  return(out)
end




"Given 3-d matrix of (independent) probability distributions get the product of them"
function probabilityProd(x::Array{T,3}) where {T<:AbstractFloat}
  J = size(x,1)
  XX = Array{T}(J, J)

  function myprod2(y::Matrix{T}, m::Int)
    Mm = size(y,1)-m+1
    if m==1
      return y[Mm,:]
    elseif m==2
      return Base.Prod2(y[Mm,:], myprod2(y, m-1))
    else
      return Base.Prod(y[Mm,:], myprod2(y, m-1))
    end
  end

  for j in 1:J
    XX[j,:] = map(prod, myprod2(x[j,:,:], size(x,2)))
  end
  return XX
end


# -------------------- big function that makes transition matrix --------------------------



"""
    makeTransitionMatrix(y::Vector{T}, S::Union{Vector{T}, Base.Iterators.AbstractProdIterator}, D::Matrix{T}, vp::VAR_process, L::Int=2, κ::T=1e-8)

Make transition matrix for discretized VAR process with `M` dimensions. The discretization
allows for each dimension to take on `Nm = length(y)` possible values. This means that there
are `Nm^M` possible states `j`. We try to match up to `L` moments.

Each row `i` of the `Nm^M` x `Nm^M` Markov transition matrix is formed from `P::Array{T}(Nm^M, M, Nm)`
by taking the Kronecker product of the `M` columns of matrix `P[i, :, :]`.

# Arguments
* `y::Vector` provides a 'generic' grid of length which is transformed into the state space
* `S::Union{Vector{T}, Base.Iterators.AbstractProdIterator}`, the primitive of the state-space
* `D::Matrix{T}` is a `M` by `Nm^M` grid of states
* `vp::VAR_process`
* `L::Int=2` the number of moments we try to match
* `κ::T=1e-8` we set the initial guess for the distribution as the max of the pdf over the
    discrete states & `κ`

# Returns
* `P::Array{T}(Nm^M, M, Nm)` matrix is probability, given we are in one state `i ∈ Nm^M`
    states that the `m ∈ M` component is equal to `j ∈ Nm`
* `JN::Matrix{T}(Nm^M, M)` is set of `min` for the objective function
* `Λ::Matrix{T}(Nm^M, M, L)` is set of `argmin`s
* `numMoments::Matrix{Int}(Nm^M, M)` number of moments successfully matched
* `approxErr::Matrix{T}(Nm^M, M, L)` approsimation error for each moment
"""
function makeTransitionMatrix(y::AbstractVector{T}, S::Union{Vector{T}, Base.Iterators.AbstractProdIterator}, D::Array{T}, vp::VAR_process{T}, L::Int=2, κ::T=1e-8) where {T<:AbstractFloat}

  if L >= length(y)
    warn("Using fewer moments. Gave ", L, " using ", length(y)-1)
    L = length(y)-1
  end

  # Initialize elements that will be returned
  Λ = zeros(T, length(S), vp.M, L)
  P = zeros(T, length(S), vp.M, length(y))
  JN = zeros(T, length(S), vp.M)
  approxErr = zeros(T, length(S), vp.M, L)
  numMoments = zeros(Int, length(S), vp.M)

  # preallocate these, which will be updated each iteration
  ΔT   = Array{T}(L,length(y))
  dev  = similar(y)
  q    = similar(y)

  δ = y[end]  # a scaling factor

  # return a Normal(0,1) PDF
  normpdf(x::Array{T,1}) = pdf(Normal(), x)

  # Conditional on each state (j), examine each of the m state vs
  prog = Progress(length(S), 5)

  for j in 1:length(S)
    for m in 1:vp.M
      dev[:] = (y - dot(vp.A[m,:], D[:,j]))
      q[:]  = max(normpdf(dev), κ)

      for l in L:-1:1
        ΔT[1:l,:] = broadcast(-, (dev / δ )'.^collect(1:l), Tbar(l, δ))

        # objective
        f(x::Array{T,1})  = dot(q,exp(x'*ΔT[1:l,:]))

        # gradient
        grad = zeros(T,l)
        function g!(x::Array{T,1}, grad::Array{T,1})
          grad[:] = 0
          for i in 1:length(q)
            grad[:] += q[i] * exp( dot(x, ΔT[1:l,i]) ) * ΔT[1:l,i]
          end
        end

        # hessian
        hess = zeros(T,l,l)
        function h!(x::Array{T,1}, hess::Array{T,2})
          hess[:,:] = 0
          for i in 1:length(q)
            hess[:,:] += q[i] * exp( dot(x, ΔT[1:l,i]) ) * (ΔT[1:l,i]*ΔT[1:l,i]')
          end
        end

        # optimize to match moments
        try
          res = Optim.optimize(f, g!, h!, Λ[j,m,1:l])
          λ = Optim.minimizer(res)
          J_candidate = Optim.minimum(res)
          g!(λ,grad)

          # if we like the results, update and break
          if ( norm( grad / λ ) < 1e-5 ) & all(isfinite(grad)) & all(isfinite(λ)) & (J_candidate > 0.0)
            JN[j,m] = J_candidate
            Λ[j,m,1:l] = λ
            for k in 1:length(y)
              P[j,m,k] = q[k] * exp( dot(λ, ΔT[1:l,k]) ) / JN[j,m]
            end
            approxErr[j,m,1:l] = grad / JN[j,m]
            numMoments[j,m] = l
            break
          end # if statment
        catch
        end


      end   # loop over moment number of conditions (l=L:1)
    end     # loop over number of states (m = 1:M)
    next!(prog)
  end       # loop over state space (j=1:J)

  return P, JN, Λ, numMoments, approxErr
end
