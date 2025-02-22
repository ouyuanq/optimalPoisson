# ADI method for solving Sylvester equation
using BandedMatrices, FillArrays
import LinearAlgebra.transpose!

# generalized ADI iteration for generalized Sylvester equation
function gadi(X0::AbstractMatrix, A1::BandedMatrix, B2::BandedMatrix, A2::BandedMatrix, B1::BandedMatrix, F::AbstractMatrix, p::AbstractVector, q::AbstractVector, tolerance::Number, tau::Integer, T1::BandedMatrix, T2::BandedMatrix)
    # generalized ADI method for  solving A_1 X B_2^T + A_2 X B_1^T = F
    # the equivalent equation is A_2\A_1 X + X (B_2\B_1)^T = A_2 \ F / B_2^T and we solve
    # (A_2\A_1 - p_j I)X_{j-1/2} = A_2 \ F / B_2^T - X_{j-1}((B_2\B_1)^T + p_j I)
    # X_j((B_2\B_1)^T + q_j I) = A_2 \ F / B_2^T - (A_2\A_1 - q_j I)X_{j-1/2}
    # multiplying both equations by A_2 on the left and B_2^T on the right to get
    # (A_1 - p_j A_2) (X_{j-1/2} B_2^T) = F - (A_2 X_{j-1})(B_1 + p_j B_2)^T
    # (A_2 X_{j})(B_1 + q_j B_2)^T = F - (A_1 - q_j A_2) (X_{j-1/2} B_2^T)
    # X0 is the initial iteration

    # set the convenient intermediate variables
    X = A2 * X0    

    # factorizations storage
    fAl, fAu = max(A1.l, A2.l), max(A1.u, A2.u)
    factorA = BandedMatrix(A1, (fAl, fAl + fAu))
    multA = BandedMatrix(A1, (fAl, fAu))
    fBl, fBu = max(B1.l, B2.l), max(B1.u, B2.u)
    factorB = BandedMatrix(B1, (fBl, fBl + fBu))
    multB = BandedMatrix(B1, (fBl, fBu))

    rhs = similar(F)
    relcauchy = one(eltype(X0))
    Xold = similar(X)
    T1XT2 = Matrix{eltype(X0)}(undef, size(T1, 1), size(T2, 1))

    for j in eachindex(p)
        # record the old iteration for checking
        if iszero(mod(j, tau))
            copyto!(Xold, X)
        end

        # shifts applications
        if j > 1
            axpy!(q[j-1]-p[j], A2, multA)
            axpy!(p[j]-q[j-1], B2, multB)
        else
            axpy!(-p[j], A2, multA)
            axpy!(p[j], B2, multB)
        end
        copyto!(factorA, multA)

        # the first step
        copyto!(rhs, F)
        bandedrmul!(rhs, X, multB)
        ldiv!(lu!(factorA), rhs)

        # shifts applications
        axpy!(p[j]-q[j], A2, multA)
        axpy!(q[j]-p[j], B2, multB)
        copyto!(factorB, multB)

        # the second step
        transpose!(copyto!(X, F))
        bandedrmul!(X, transpose!(rhs), multA)
        ldiv!(lu!(factorB), X)
        transpose!(X)

        # check relative incremental error
        if iszero(mod(j, tau))
            # ||X_j - X_{j-1}||
            axpy!(-1, X, Xold)
            # compute the coefficients in recombined basis
            bandedldiv!(A2, Xold)
            copyto!(view(T1XT2, 1:size(Xold, 1), 1:size(Xold, 2)), Xold)
            # convert back to Chebyshev coefficients
            bandedlmul!(T1, T1XT2)
            transpose!(T1XT2)
            bandedlmul!(T2, T1XT2)
            temp = norm(T1XT2)

            # ||X_j||
            copyto!(Xold, X)
            # compute the coefficients in recombined basis
            bandedldiv!(A2, Xold)
            copyto!(view(T1XT2, 1:size(Xold, 1), 1:size(Xold, 2)), Xold)
            # convert back to Chebyshev coefficients
            bandedlmul!(T1, T1XT2)
            transpose!(T1XT2)
            bandedlmul!(T2, T1XT2)
            temp = temp / norm(T1XT2)

            # @printf "now %.3e and old %.3e \n" temp relcauchy

            # if the relative error is small than the tolerance or stagnates, we break
            if temp < tolerance || temp > relcauchy
                # @printf "executed %i and total %i \n" j length(p)
                break
            end

            relcauchy = temp
        end
    end

    # convert back to X
    bandedldiv!(A2, X)

    X
end

# generalized ADI with relative incremental and true error during iterations
function gadi_iter(X0::AbstractMatrix, A1::BandedMatrix, B2::BandedMatrix, A2::BandedMatrix, B1::BandedMatrix, F::AbstractMatrix, p::AbstractVector, q::AbstractVector, tolerance::Number, T1::BandedMatrix, T2::BandedMatrix; truesol::AbstractMatrix=Zeros(0, 0))
    # same as gadi but Cauchy error are recorded every iteration

    # set the convenient intermediate variables
    X = A2 * X0

    # factorizations storage
    fAl, fAu = max(A1.l, A2.l), max(A1.u, A2.u)
    factorA = BandedMatrix(A1, (fAl, fAl + fAu))
    multA = BandedMatrix(A1, (fAl, fAu))
    fBl, fBu = max(B1.l, B2.l), max(B1.u, B2.u)
    factorB = BandedMatrix(B1, (fBl, fBl + fBu))
    multB = BandedMatrix(B1, (fBl, fBu))

    rhs = similar(F)
    cauchyvec = zeros(eltype(X0), length(p))
    truevec = zeros(eltype(X0), length(p))
    Xold = similar(X)
    T1XT2 = Matrix{eltype(X0)}(undef, size(T1, 1), size(T2, 1))

    # transpose!(X) # for convenience
    for j in eachindex(p)
        # record the old iteration for checking
        copyto!(Xold, X)

        # shifts applications
        if j > 1
            axpy!(q[j-1]-p[j], A2, multA)
            axpy!(p[j]-q[j-1], B2, multB)
        else
            axpy!(-p[j], A2, multA)
            axpy!(p[j], B2, multB)
        end
        copyto!(factorA, multA)

        # the first step
        copyto!(rhs, F)
        bandedrmul!(rhs, X, multB)
        ldiv!(lu!(factorA), rhs)

        # shifts applications
        axpy!(p[j]-q[j], A2, multA)
        axpy!(q[j]-p[j], B2, multB)
        copyto!(factorB, multB)

        # the second step
        transpose!(copyto!(X, F))
        bandedrmul!(X, transpose!(rhs), multA)
        ldiv!(lu!(factorB), X)
        transpose!(X)

        # # check relative incremental error
        # T1XT2 = T1 * (A2 \ (X - Xold)) * transpose(T2)
        # cauchyvec[j] = norm(T1XT2)

        # # ||X_j||
        # T1XT2 = T1 * (A2 \ X) * transpose(T2)
        # cauchyvec[j] = cauchyvec[j] / norm(T1XT2)

        # # relative true error
        # axpy!(-1, truesol, view(T1XT2, 1:size(truesol, 1), 1:size(truesol, 2)))
        # truevec[j] = norm(T1XT2) / norm(truesol)
        
        # check relative incremental error
        axpy!(-1, X, Xold)
        if A2.l == 0
            bandedldiv!(A2, Xold)
        else
            copyto!(Xold, A2 \ Xold)
        end
        bandedlrmul!(T1XT2, T1, Xold, T2)
        cauchyvec[j] = norm(T1XT2)

        # ||X_j||
        copyto!(Xold, X)
        if A2.l == 0
            bandedldiv!(A2, Xold)
        else
            copyto!(Xold, A2 \ Xold)
        end
        bandedlrmul!(T1XT2, T1, Xold, T2)
        cauchyvec[j] = cauchyvec[j] / norm(T1XT2)

        # relative true error
        axpy!(-1, truesol, view(T1XT2, 1:size(truesol, 1), 1:size(truesol, 2)))
        truevec[j] = norm(T1XT2) / norm(truesol)

        # if cauchyvec[j] < tolerance
        #     break
        # end
    end

    # convert back to X
    X = A2 \ X

    X, cauchyvec, truevec
end

@inline function transpose!(A::AbstractMatrix{T}) where T<:Number
    # transpose the matrix in place
    m, n = size(A)

    @assert m == n "Cannot transpose a rectangular matrix"
    @inbounds @simd for j = 1:n-1
        for i = j+1:n
            A[i, j], A[j, i] = A[j, i], A[i, j]
        end
    end

    A
end

## ADI iterations for Sylvester equation
function adi(X0::AbstractMatrix, A::BandedMatrix, B::BandedMatrix, F::AbstractMatrix, p::AbstractVector, q::AbstractVector)
    # ADI method for  solving A X + X B^T = F
    # the iterations are
    # (A - p_j I) X_{j-1/2} = F - X_{j-1} (B + p_j I)^T
    # X_{j} (B + q_j I)^T = F - (A - q_j I) X_{j-1/2}
    # X0 is the initial iteration and p and q are shifts

    # factorizations storage
    factorA = BandedMatrix(A, (A.l, A.l + A.u))
    factorB = BandedMatrix(B, (B.l, B.l + B.u))
    Ia = BandedMatrix(0=>Ones(Bool, size(A, 1)))
    Ib = BandedMatrix(0=>Ones(Bool, size(B, 1)))

    rhs = similar(F)
    X = Matrix(X0)
    for j in eachindex(p)
        # shifts applications
        copyto!(factorA, A)
        axpy!(-p[j], Ia, factorA)

        # the first step
        copyto!(rhs, F)
        bandedrmul!(rhs, X, B)
        axpy!(-p[j], X, rhs)
        ldiv!(qr!(factorA), rhs)

        # shifts applications
        copyto!(factorB, B)
        axpy!(q[j], Ib, factorB)

        # the second step
        transpose!(copyto!(X, F))
        bandedrmul!(X, transpose!(rhs), A)
        axpy!(q[j], rhs, X)
        ldiv!(qr!(factorB), X)
        transpose!(X)
    end

    X
end

# ## GADI iterations for TO method
# function gadi(X0::AbstractMatrix, A1::AlmostBandedMatrix, B2::AlmostBandedMatrix, A2::AlmostBandedMatrix, B1::AlmostBandedMatrix, F::AbstractMatrix, p::AbstractVector, q::AbstractVector)
#     # generalized ADI method for  solving A_1 X B_2^T + A_2 X B_1^T = F
#     # the equivalent equation is A_2\A_1 X + X (B_2\B_1)^T = A_2 \ F / B_2^T and we solve
#     # (A_2\A_1 - p_j I)X_{j-1/2} = A_2 \ F / B_2^T - X_{j-1}((B_2\B_1)^T + p_j I)
#     # X_j((B_2\B_1)^T + q_j I) = A_2 \ F / B_2^T - (A_2\A_1 - q_j I)X_{j-1/2}
#     # multiplying both equations by A_2 on the left and B_2^T on the right to get
#     # (A_1 - p_j A_2) (X_{j-1/2} B_2^T) = F - (A_2 X_{j-1})(B_1 + p_j B_2)^T
#     # (A_2 X_{j})(B_1 + q_j B_2)^T = F - (A_1 - q_j A_2) (X_{j-1/2} B_2^T)
#     # X0 is the initial iteration

#     # set the convenient intermediate variables
#     X = A2 * X0

#     # factorizations storage
#     fAl, fAu = max(A1.bands.l, A2.bands.l), max(A1.bands.u, A2.bands.u)
#     factorA = AlmostBandedMatrix(A1, (fAl, fAl + fAu))
#     multA = AlmostBandedMatrix(A1, (fAl, fAu))
#     fBl, fBu = max(B1.bands.l, B2.bands.l), max(B1.bands.u, B2.bands.u)
#     factorB = AlmostBandedMatrix(B1, (fBl, fBl + fBu))
#     multB = AlmostBandedMatrix(B1, (fBl, fBu))

#     rhs = similar(F)
#     for j in eachindex(p)
#         # shifts applications
#         if j > 1
#             axpy!(q[j-1]-p[j], A2.bands, multA.bands)
#             axpy!(q[j-1]-p[j], A2.fill.args[2], multA.fill.args[2])
#             axpy!(p[j]-q[j-1], B2.bands, multB.bands)
#             axpy!(p[j]-q[j-1], B2.fill.args[2], multB.fill.args[2])
#         else
#             axpy!(-p[j], A2.bands, multA.bands)
#             axpy!(-p[j], A2.fill.args[2], multA.fill.args[2])
#             axpy!(p[j], B2.bands, multB.bands)
#             axpy!(p[j], B2.fill.args[2], multB.fill.args[2])
#         end
#         copyto!(factorA, multA)
#         fillgaps!(factorA, multA)

#         # the first step
#         copyto!(rhs, F)
#         almostbandedrmul!(rhs, X, multB)
#         ldiv!(qr!(factorA), rhs)

#         # shifts applications
#         axpy!(p[j]-q[j], A2.bands, multA.bands)
#         axpy!(p[j]-q[j], A2.fill.args[2], multA.fill.args[2])
#         axpy!(q[j]-p[j], B2.bands, multB.bands)
#         axpy!(q[j]-p[j], B2.fill.args[2], multB.fill.args[2])
#         copyto!(factorB, multB)
#         fillgaps!(factorB, multB)

#         # the second step
#         transpose!(copyto!(X, F))
#         almostbandedrmul!(X, transpose!(rhs), multA)
#         ldiv!(qr!(factorB), X)
#         transpose!(X)
#     end

#     # convert back to X
#     X = A2 \ X

#     X
# end