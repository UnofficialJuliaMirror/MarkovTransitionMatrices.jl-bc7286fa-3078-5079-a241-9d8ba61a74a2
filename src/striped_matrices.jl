# -------------------- make striped versions --------------------------


"""
    striped_bool(m::Integer, n::Integer, ur::UnitRange)::Matrix{Bool}(m,n)

Returns striped array where trues are on diagonals `ur`
"""
function striped_bool(m::Integer, n::Integer, ur::UnitRange)::Matrix{Bool}
  out   = fill(false, m, n)
  trues = fill(true, m, n)
  for i in ur
    out .+= diagm(diag(trues, i), i)
  end
  return out
end


"""
    stripe_and_sparsify_transition_matrix{T<:AbstractFloat}(P::Matrix{T}, minp::T, which_diags::UnitRange)::SparseMatrixCSC{T}

Construct sparse transition matrix from `P` where elements not on diagonals `which_diags`
or less than `minp` are zeroed out. (Ensures that rows sum to 1)
"""
function sparsify_transition_matrix{T<:AbstractFloat}(P::Matrix{T}, minp::T, which_diags::UnitRange)::SparseMatrixCSC{T}

  sb = striped_bool(size(P, 1), size(P, 2), which_diags)
  P_big = P .> minp
  P_new = P .* sb .* P_big
  P_new_rowsums = sum(P_new, 2)
  for i in 1:size(P_new, 1)
    P_new[i, :] .= P_new[i, :] ./ P_new_rowsums[i]
  end

  return sparse(P_new)

end


function sparsify_transition_matrix{T<:AbstractFloat}(P::Matrix{T}, minp::T)::SparseMatrixCSC{T}

  P_big = P .> minp
  P_new = P .* P_big
  P_new_rowsums = sum(P_new, 2)
  for i in 1:size(P_new, 1)
    P_new[i, :] .= P_new[i, :] ./ P_new_rowsums[i]
  end

  return sparse(P_new)

end
