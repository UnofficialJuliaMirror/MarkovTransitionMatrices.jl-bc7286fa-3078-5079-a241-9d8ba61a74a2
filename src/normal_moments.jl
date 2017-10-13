import Base: start, next, done, length

abstract type AbstractNormCentralMoment end

struct stdNormCentralMoment <: AbstractNormCentralMoment
  n::Int
end

struct NormCentralMoment{T<:Real} <: AbstractNormCentralMoment
  n::Int
  σ::T
end

Base.done(  S::AbstractNormCentralMoment, state::Tuple) = state[1] > S.n
Base.length(S::AbstractNormCentralMoment) = S.n
Base.start(  ::AbstractNormCentralMoment) = (1, 1.0,)

oddmoment(S::stdNormCentralMoment, nf::Real, i::Int) = nf
oddmoment(S::NormCentralMoment   , nf::Real, i::Int) = nf * S.σ^i

Base.getindex(S::NormCentralMoment   , k::Integer) = iseven(k) ? 0.0 : 2^(-k/2) * factorial(k) / factorial(k/2) * S.σ^k
Base.getindex(S::stdNormCentralMoment, k::Integer) = iseven(k) ? 0.0 : 2^(-k/2) * factorial(k) / factorial(k/2) * 1.0

function Base.next(S::AbstractNormCentralMoment, state::Tuple)
  i, nf = state
  isodd(i) && return (0., (i+1, nf))
  nf *= (i-1)
  return (oddmoment(S,nf,i), (i+1, nf))
end
