# factored ADI iteration
# see  P.Benner, R.-C.Li, and N.Truhar, On the ADI method for Sylvester equations, Journal of Computational and Applied Mathematics, 233(2009), pp.10351045.

function gfadi(A1::BandedMatrix, B2::BandedMatrix, A2::BandedMatrix, B1::BandedMatrix, UF::AbstractMatrix, VF::AbstractMatrix, p::AbstractVector, q::AbstractVector, tolerance::Number, tau::Integer)
    # generalized factored ADI method for solving A_1 X B_2^T + A_2 X B_1^T = F
    # the equivalent equation is A_2\A_1 X + X (B_2\B_1)^T = A_2 \ (UF * VF^T) / B_2^T and we solve
    # Z_1 = (A_1 - p_1 A_2) \ UF
    # Z_{j+1} = Z_{j} + (p_{j+1} - q_{j})(A_1 - p_{j+1} A_2) \ (A_2 * Z_{j})
    # Y_1 = (B_1 + q_1 B_2) \ VF
    # Y_{j+1} = Y_{j} + (p_{j} - q_{j+1})(B_1 + q_{j+1} B_2) \ (B_2 * Y_{j})
    # so that iteration X_k = (Z_1 Z_2 ... Z_k) diag((q_1 - p_1), (q_2 - p_2), ... (q_k - p_k)) (Y_1 Y_2 ... Y_k)^T 
    # X0 is the initial iteration

    # set the full solution (can be removed if no error check is needed)
    X = zeros(eltype(UF), size(A1, 2), size(B2, 2))

    # factorizations storage
    fAl, fAu = max(A1.l, A2.l), max(A1.u, A2.u)
    factorA = BandedMatrix(A1, (fAl, fAl + fAu))
    fBl, fBu = max(B1.l, B2.l), max(B1.u, B2.u)
    factorB = BandedMatrix(B1, (fBl, fBl + fBu))

    # factors for X_j
    Zvec = Vector{typeof(UF)}(undef, 0)
    Yvec = Vector{typeof(VF)}(undef, 0)
    Dvec = Vector{eltype(p)}(undef, 0)

    relcauchy = one(eltype(UF))
    for j in eachindex(p)
        # shifts applications
        if j > 1
            copyto!(factorA, A1)
            copyto!(factorB, B1)
            axpy!(-p[j], A2, factorA)
            axpy!(q[j], B2, factorB)
            Znow = A2 * Zvec[end]
            Ynow = B2 * Yvec[end]
        else
            axpy!(-p[j], A2, factorA)
            axpy!(q[j], B2, factorB)
            Znow = copy(UF)
            Ynow = copy(VF)
        end

        # division
        ldiv!(lu!(factorA), Znow)
        ldiv!(lu!(factorB), Ynow)

        if j > 1
            axpby!(true, Zvec[end], p[j] - q[j-1], Znow)
            axpby!(true, Yvec[end], p[j-1] - q[j], Ynow)
        end

        # store the factors
        push!(Zvec, Znow)
        push!(Yvec, Ynow)
        push!(Dvec, q[j] - p[j])

        # for error check
        mul!(X, Znow, transpose(Ynow), Dvec[j], true)

        # check relative Cauchy error
        if iszero(mod(j, tau))
            temp = Dvec[j] * norm(Znow * Ynow')

            # @printf "now %.3e and old %.3e \n" temp relcauchy

            # if the relative error is small than the tolerance or stagnates, we break
            if temp < tolerance || temp > relcauchy
                # @printf "executed %i and total %i \n" j length(p)
                break
            end

            relcauchy = temp
        end
    end

    X
end

# # not economic, for test only
# function gfadi2(A1::BandedMatrix, B2::BandedMatrix, A2::BandedMatrix, B1::BandedMatrix, UF::AbstractMatrix, VF::AbstractMatrix, p::AbstractVector, q::AbstractVector, tolerance::Number, tau::Integer)
#     # generalized factored ADI method for solving A_1 X B_2^T + A_2 X B_1^T = F
#     # the equivalent equation is A_2\A_1 X + X (B_2\B_1)^T = A_2 \ (UF * VF^T) / B_2^T and we solve
#     # Z_1 = (A_1 - p_1 A_2) \ UF
#     # Z_{j+1} = Z_{j} + (p_{j+1} - q_{j})(A_1 - p_{j+1} A_2) \ (A_2 * Z_{j})
#     # Y_1 = (B_1 + q_1 B_2) \ VF
#     # Y_{j+1} = Y_{j} + (p_{j} - q_{j+1})(B_1 + q_{j+1} B_2) \ (B_2 * Y_{j})
#     # so that iteration X_k = (Z_1 Z_2 ... Z_k) diag((q_1 - p_1), (q_2 - p_2), ... (q_k - p_k)) (Y_1 Y_2 ... Y_k)^T 
#     # X0 is the initial iteration

#     # set the full solution (can be removed if no error check is needed)
#     X = zeros(eltype(UF), size(A1, 2), size(B2, 2))

#     # factorizations storage
#     fAl, fAu = max(A1.l, A2.l), max(A1.u, A2.u)
#     factorA = BandedMatrix(A1, (fAl, fAl + fAu))
#     fBl, fBu = max(B1.l, B2.l), max(B1.u, B2.u)
#     factorB = BandedMatrix(B1, (fBl, fBl + fBu))

#     # factors for X_j
#     Zvec = Vector{typeof(UF)}(undef, 0)
#     Yvec = Vector{typeof(VF)}(undef, 0)
#     Dvec = Vector{eltype(p)}(undef, 0)

#     relcauchy = one(eltype(UF))
#     for j in eachindex(p)
#         if j > 1
#             copyto!(factorA, A1)
#             copyto!(factorB, B1)
#             axpy!(-q[j-1], A2, factorA)
#             axpy!(p[j-1], B2, factorB)
#             Znow = factorA * Zvec[end]
#             Ynow = factorB * Yvec[end]
#             axpy!(q[j-1] - p[j], A2, factorA)
#             axpy!(q[j] - p[j-1], B2, factorB)
#         else
#             axpy!(-p[j], A2, factorA)
#             axpy!(q[j], B2, factorB)
#             Znow = copy(UF)
#             Ynow = copy(VF)
#         end

#         # division
#         ldiv!(lu!(factorA), Znow)
#         ldiv!(lu!(factorB), Ynow)

#         # store the factors
#         push!(Zvec, Znow)
#         push!(Yvec, Ynow)
#         push!(Dvec, q[j] - p[j])

#         # for error check
#         mul!(X, Znow, transpose(Ynow), Dvec[j], true)

#         # check relative Cauchy error
#         if iszero(mod(j, tau))
#             temp = Dvec[j] * norm(Znow * Ynow')

#             # @printf "now %.3e and old %.3e \n" temp relcauchy

#             # if the relative error is small than the tolerance or stagnates, we break
#             if temp < tolerance || temp > relcauchy
#                 # @printf "executed %i and total %i \n" j length(p)
#                 break
#             end

#             relcauchy = temp
#         end
#     end

#     X
# end