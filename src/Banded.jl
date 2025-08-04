# auxiliary functions related to banded matrices
using BandedMatrices
using BandedMatrices: AbstractBandedMatrix, bandeddata, _banded_square_ldiv!
import LinearAlgebra.rdiv!

# left multiplication of a triangular banded matrix
function bandedlmul!(A::BandedMatrix, B::AbstractVecOrMat)
    @assert size(A, 1) == size(B, 1) "Unmatched dimension of A and B"
    @assert size(A, 1) >= size(A, 2) "Short fat A is not supported"
    @assert A.l * A.u <= 0 "Nontriangular banded matrix A"

    temp = zero(promote_type(eltype(A), eltype(B)))
    if A.l <= 0
        # uppertriangular
        @inbounds for k in axes(B, 2)
            for i in axes(A, 1)
                temp = 0
                for j in rowrange(A, i)
                    temp += BandedMatrices.inbands_getindex(A, i, j) * B[j, k]
                end
                B[i, k] = temp
            end
        end
    elseif A.u <= 0
        # lowertriangular
        @inbounds for k in axes(B, 2)
            for i in reverse(axes(A, 1))
                temp = 0
                for j in rowrange(A, i)
                    temp += BandedMatrices.inbands_getindex(A, i, j) * B[j, k]
                end
                B[i, k] = temp
            end
        end
    end

    B
end

function bandedlrmul!(C::AbstractMatrix, A1::BandedMatrix, B::AbstractMatrix, A2::BandedMatrix)
    # compute C = A1 * B * transpose(A2)
    @assert size(A1, 2) == size(B, 1) && size(A2, 2) == size(B, 2) "Unmatched dimension of B"
    @assert size(A1, 1) == size(C, 1) && size(A2, 1) == size(C, 2) "Unmatched dimension of C"
    @assert A1.l * A1.u <= 0 && A2.l * A2.u <= 0 "Nontriangular banded matrix"
    @assert size(C, 1) >= size(B, 1) && size(C, 2) >= size(B, 2) "C should be larger than B"

    # C = B * transpose(A2)
    szB1 = size(B, 1)
    @inbounds for j in axes(C, 2)
        Cj = view(C, 1:szB1, j) .= 0
        for k in rowrange(A2, j)
            axpy!(BandedMatrices.inbands_getindex(A2, j, k), view(B, :, k), Cj)
        end
    end

    # C = A1 * C
    temp = zero(promote_type(eltype(C), eltype(B)))
    # A1 * B
    if A1.l <= 0
        # uppertriangular
        @inbounds for k in axes(C, 2)
            for i in axes(A1, 1)
                temp = 0
                for j in rowrange(A1, i)
                    temp += BandedMatrices.inbands_getindex(A1, i, j) * C[j, k]
                end
                C[i, k] = temp
            end
        end
    elseif A1.u <= 0
        # lowertriangular
        @inbounds for k in axes(C, 2)
            for i in reverse(axes(A1, 1))
                temp = 0
                for j in rowrange(A1, i)
                    temp += BandedMatrices.inbands_getindex(A1, i, j) * C[j, k]
                end
                C[i, k] = temp
            end
        end
    end

    C
end

# # left and right multiplication of lowertriangular banded matrix
# function bandedlrmul!(C::AbstractMatrix, A1::BandedMatrix, B::AbstractMatrix, A2::BandedMatrix)
#     # compute C = A1 * B * transpose(A2)
#     @assert size(A1, 2) == size(B, 1) && size(A2, 2) == size(B, 2) "Unmatched dimension of B"
#     @assert size(A1, 1) == size(C, 1) && size(A2, 1) == size(C, 2) "Unmatched dimension of C"
#     @assert A1.l * A1.u <= 0 && A2.l * A2.u <= 0 "Nontriangular banded matrix"
#     @assert size(C, 1) >= size(B, 1) && size(C, 2) >= size(B, 2) "C should be larger than B"

#     temp = zero(promote_type(eltype(C), eltype(B)))

#     # A1 * B
#     if A1.l <= 0
#         # uppertriangular
#         @inbounds for k in axes(B, 2)
#             for i in axes(A1, 1)
#                 temp = 0
#                 for j in rowrange(A1, i)
#                     temp += BandedMatrices.inbands_getindex(A1, i, j) * B[j, k]
#                 end
#                 C[i, k] = temp
#             end
#         end
#     elseif A1.u <= 0
#         # lowertriangular
#         @inbounds for k in axes(B, 2)
#             for i in reverse(axes(A1, 1))
#                 temp = 0
#                 for j in rowrange(A1, i)
#                     temp += BandedMatrices.inbands_getindex(A1, i, j) * B[j, k]
#                 end
#                 C[i, k] = temp
#             end
#         end
#     end

#     # A2 * transpose(C)
#     if A2.l <= 0
#         # uppertriangular
#         @inbounds for k in axes(C, 1)
#             for i in axes(A2, 1)
#                 temp = 0
#                 for j in rowrange(A2, i)
#                     temp += BandedMatrices.inbands_getindex(A2, i, j) * C[k, j]
#                 end
#                 C[k, i] = temp
#             end
#         end
#     elseif A2.u <= 0
#         # lowertriangular
#         @inbounds for k in axes(C, 1)
#             for i in reverse(axes(A2, 1))
#                 temp = 0
#                 for j in rowrange(A2, i)
#                     temp += BandedMatrices.inbands_getindex(A2, i, j) * C[k, j]
#                 end
#                 C[k, i] = temp
#             end
#         end
#     end

#     C
# end

# left division of a triangular banded matrix
function bandedldiv!(A::BandedMatrix, B::AbstractVecOrMat)
    @assert size(A, 2) == size(B, 1) "Unmatched dimension of A and B"
    @assert size(A, 1) == size(A, 2) "Nonsquare A is not supported"
    @assert A.l * A.u == 0 "Nontriangular or singular banded matrix A"

    if A.l == 0
        # uppertriangular
        @inbounds for k in axes(B, 2)
            for i in reverse(axes(A, 1))
                for j in i+1:rowrange(A, i)[end]
                    B[i, k] -= BandedMatrices.inbands_getindex(A, i, j) * B[j, k]
                end
                B[i, k] = BandedMatrices.inbands_getindex(A, i, i) \ B[i, k]
            end
        end
    elseif A.u == 0
        # lowertriangular
        @inbounds for k in axes(B, 2)
            for i in axes(A, 1)
                for j in rowrange(A, i)[1]:i-1
                    B[i, k] -= BandedMatrices.inbands_getindex(A, i, j) * B[j, k]
                end
                B[i, k] = BandedMatrices.inbands_getindex(A, i, i) \ B[i, k]
            end
        end
    end

    B
end

function bandedlmul!(C::AbstractMatrix, A::BandedMatrix, B::AbstractMatrix)
    # specilized function for multiplication of banded matrix and dense matrix used in ADI iterations
    # C = C - A*B where A is a banded matrix and has nonzero elements only on even diagonals

    Adata = A.data
    lA, uA = bandwidths(A)
    szA1, szA2 = size(A)

    @inbounds for d = 1:lA+uA+1
        rowstart = max(uA-d+2, 1)
        rowstop = szA2 - max(d-uA-1, 0)
        colstart = max(d-uA, 1)
        colstop = szA1 - max(uA-d+1, 0)
        dA = Diagonal(view(Adata, d, max((uA-d+2), 1).+(0:rowstop-rowstart)))
        mul!(view(C, colstart:colstop, :), dA, view(B, rowstart:rowstop, :), -1, true)
    end

    C
end

@inline function mybandedlmul!(C::AbstractMatrix{T}, A::BandedMatrix{T}, B::AbstractMatrix{T}) where T
    # specilized function for multiplication of dense matrix and transpose of banded matrix used in ADI iterations
    # C = C - B*A^T where A is a banded matrix
    n = size(A, 1)
    Al, Au = bandwidths(A)

    @assert size(C, 2) == size(B, 2) && size(A, 2) == size(B, 1) "DimensionMismatch"

    @inbounds @simd for j in axes(C, 2)
        Cj = view(C, :, j)
        Bj = view(B, :, j)
        LinearAlgebra.BLAS.gbmv!('N', n, Al, Au, -one(T), A.data, Bj, true, Cj)
    end

    C
end

# function bandedrmul!(C::AbstractMatrix, B::AbstractMatrix, A::BandedMatrix)
#     # specilized function for multiplication of dense matrix and transpose of banded matrix used in ADI iterations
#     # C = C - B*A^T where A is a banded matrix

#     Adata = A.data
#     lA, uA = bandwidths(A)
#     szA1, szA2 = size(A)

#     @inbounds for d = 1:lA+uA+1
#         rowstart = max(uA-d+2, 1)
#         rowstop = szA2 - max(d-uA-1, 0)
#         colstart = max(d-uA, 1)
#         colstop = szA1 - max(uA-d+1, 0)
#         dA = Diagonal(view(Adata, d, max((uA-d+2), 1).+(0:rowstop-rowstart)))
#         mul!(view(C, :, colstart:colstop), view(B, :, rowstart:rowstop), dA, -1, true)
#     end

#     C
# end

function bandedrmul!(C::AbstractMatrix, B::AbstractMatrix, A::BandedMatrix)
    # specilized function for multiplication of dense matrix and transpose of banded matrix used in ADI iterations
    # C = C - B*A^T where A is a banded matrix

    @inbounds for j in axes(C, 2)
        Cj = view(C, :, j)
        for k in rowrange(A, j)
            axpy!(-BandedMatrices.inbands_getindex(A, j, k), view(B, :, k), Cj)
            # axpy!(-A[j, k], view(B, :, k), Cj)
        end
    end

    C
end

# function almostbandedlmul(A::AbstractMatrix, B::AbstractMatrix)
#     # specilized function for multiplication of almostbanded matrix and dense matrix, i.e., A * B where A is an almostbanded matrix

#     # banded part
#     Ab = A.bands
#     AB = Ab * B
#     # low-rank part for A * B
#     Af = A.fill.args
#     Af2B = similar(B, size(Af[2], 1), size(B, 2))
#     for i in axes(Af[2], 1)
#         rowind = rowrange(Ab, i)[end]+1:size(Af[2], 2)
#         mul!(view(Af2B, i:i, :), view(Af[2], i:i, rowind), view(B, rowind, :), true, false)
#     end
#     mul!(AB, Af[1], Af2B, true, true)

#     AB
# end

# function almostbandedrmul!(C::AbstractMatrix, B::AbstractMatrix, A::AlmostBandedMatrix)
#     # specilized function for multiplication of dense matrix and transpose of banded matrix used in ADI iterations
#     # C = C - B*A^T where A is an almostbanded matrix

#     # banded part
#     Ab = A.bands
#     @inbounds for j in axes(C, 2)
#         Cj = view(C, :, j)
#         for k in rowrange(Ab, j)
#             axpy!(-Ab[j, k], view(B, :, k), Cj)
#         end
#     end

#     # low-rank part
#     a = A.fill.args[1][1] # the multiplier of fill matrix in B
#     Af = A.fill.args[2]
#     lr = minimum(size(Af))
#     @inbounds for j = 1:lr
#         Cj = view(C, :, j)
#         for k = rowrange(Ab, j)[end]+1:size(A, 2)
#             axpy!(-Af[j, k] * a, view(B, :, k), Cj)
#         end
#     end

#     C
# end

# function almostbandedlrmul(A::AlmostBandedMatrix, B::AbstractMatrix, C::AlmostBandedMatrix)
#     # specilized function for multiplication of almostbanded matrix, dense matrix and transpose of almostbanded matrix, i.e., compute A * B * C^T where A and C are almostbanded matrix

#     # banded part for A * B
#     Ab = A.bands
#     AB = Ab * B
#     # low-rank part for A * B
#     Af = A.fill.args
#     Af2B = similar(B, size(Af[2], 1), size(B, 2))
#     for i in axes(Af[2], 1)
#         rowind = rowrange(Ab, i)[end]+1:size(Af[2], 2)
#         mul!(view(Af2B, i:i, :), view(Af[2], i:i, rowind), view(B, rowind, :), true, false)
#     end
#     mul!(AB, Af[1], Af2B, true, true)

#     # banded part for AB * C^T
#     Cb = C.bands
#     ABCt = AB * transpose(Cb)
#     # low-rank part for AB * C^T
#     Cf = C.fill.args
#     ABCf2t = similar(B, size(AB, 1), size(Cf[2], 1))
#     for i in axes(Cf[2], 1)
#         rowind = rowrange(Cb, i)[end]+1:size(Cf[2], 2)
#         mul!(view(ABCf2t, :, i), view(AB, :, rowind), view(Cf[2], i, rowind), true, false)
#     end
#     mul!(ABCt, ABCf2t, transpose(Cf[1]), true, true)

#     ABCt
# end

# function fillgaps!(A::AlmostBandedMatrix, B::AlmostBandedMatrix)
#     # fill the banded part of A by low rank part of B
#     a = B.fill.args[1][1]  # the multiplier of fill matrix in B
#     for i in axes(B.fill.args[2], 1)
#         for j = rowrange(B.bands, i)[end]+1:rowrange(A.bands, i)[end]
#             A.bands[i, j] = a * B.fill.args[2][i, j]
#         end
#     end
# end

# Kronecker product of banded matrix
function bandedkron(A::BandedMatrix, B::BandedMatrix)
    K = BandedMatrix{T}(undef, (size(A, 1) * size(B, 1), size(A, 2) * size(B, 2)), (bandwidth(A, 1) * size(B, 1) + bandwidth(B, 1), bandwidth(A, 2) * size(B, 2) + bandwidth(B, 2)))

    @inbounds for jA in axes(A, 2)
        for iA in colrange(A, jA)
            axpby!(A[iA, jA], B, false, view(K, (iA-1)*size(B, 1)+1:iA*size(B, 1), (jA-1)*size(B, 2)+1:jA*size(B, 2)))
        end
    end

    K
end


# function myldiv!(A::Factorization{T}, B::AbstractMatrix{T}) where {T<:AbstractFloat}
#     @inbounds for j in axes(B, 2)
#         ldiv!(A, view(B, :, j))
#     end
#     B
# end

# function rdiv!(B::AbstractMatrix{T}, A::QR{T, U}) where {T, U<:AbstractBandedMatrix}
#     @assert size(A, 1) == size(A, 2) == size(B, 2)

#     R = A.factors
#     rmul!(B, A.Q)
#     B .= Rdiv(B, transpose(UpperTriangular(R)))

#     B
# end

# function myrdiv!(B::AbstractMatrix{T}, A::QR{T, U}) where {T, U<:AbstractBandedMatrix}
#     @assert size(A, 1) == size(A, 2) == size(B, 2)

#     @inbounds for j in axes(B, 1)
#         rdiv!(view(B, j:j, :), A)
#     end
#     B

#     B
# end

# function myrdiv2!(B::AbstractMatrix{T}, A::QR{T, U}) where {T, U<:AbstractBandedMatrix}
#     @assert size(A, 1) == size(A, 2) == size(B, 2)

#     @inbounds for j in axes(B, 1)
#         _banded_square_ldiv!(A, view(transpose(B), :, j))
#     end
#     B

#     B
# end