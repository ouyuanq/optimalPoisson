# initialization for Townsend & Olver's  method

function poissonTO_mixedbc_init(::Type{T}, m::Integer, n::Integer, f::Function, lbc::Function, rbc::Function, dbc::Function, ubc::Function, theta::Number) where T
    # initialization of Townsend & Olver's method for solving Poisson equation u_xx + u_yy = f with left Dirichlet, right Neumann, down Dirichlet and upper Robin conditions
    N = 2

    # set up the operators
    Sy = convertmat(T, m-N, m, 0, N)
    Dy = diffmat(T, m-N, m, N)

    if n == m
        # save for the same dimension of x- and y-direction
        Sx, Dx = Sy, Dy
    else
        Sx = convertmat(T, n-N, n, 0, N)
        Dx = diffmat(T, n-N, n, N)
    end

    # rhs
    F = Sy * coeffs2(f, m-1, n-1, T) * transpose(Sx)

    # construction for down and upper boundary conditions 
    bcy = Matrix{T}(undef, 2, m)
    bcy[1, 1:2:end] .= 1
    bcy[1, 2:2:end] .= -1
    broadcast!(x -> theta * x^2, view(bcy, 2, :), 0:m-1)
    axpy!(true, Ones(T, m), view(bcy, 2, :))
    H = zeros(T, 2, n)
    copyto!(view(H, 1, :), coeffs(dbc, T))
    copyto!(view(H, 2, :), coeffs(ubc, T))
    # elimination
    H = bcy[1:2, 1:2] \ H
    bcy = bcy[1:2, 1:2] \ bcy
    mSy = mul!(Matrix(Sy), view(Sy, :, 1:2), bcy, -1, true)
    # mSy = AlmostBandedMatrix(BandedMatrix(view(mSy, :, 3:m), (2, 2)), ApplyMatrix(*, Matrix{T}(I, m-N, N), view(mSy, 1:N, 3:m)))
    # mDy = BandedMatrix(view(Dy, :, 3:m), (0, 0))
    mSy = view(mSy, :, 3:m)
    mDy = view(Dy, :, 3:m)
    mul!(F, view(Sy, :, 1:2), H*transpose(Dx), -1, true)

    # construction for left and right boundary conditions 
    bcx = Matrix{T}(undef, 2, n)
    bcx[1, 1:2:end] .= 1
    bcx[1, 2:2:end] .= -1
    broadcast!(x -> x^2, view(bcx, 2, :), 0:n-1)
    G = zeros(T, 2, m)
    copyto!(view(G, 1, :), coeffs(lbc, T))
    copyto!(view(G, 2, :), coeffs(rbc, T))
    # elimination
    G = bcx[1:2, 1:2] \ G
    bcx = bcx[1:2, 1:2] \ bcx
    mSx = mul!(Matrix(Sx), view(Sx, :, 1:2), bcx, -1, true)
    # mSx = AlmostBandedMatrix(BandedMatrix(view(mSx, :, 3:n), (2, 2)), ApplyMatrix(*, Matrix{T}(I, n-N, N), view(mSx, 1:N, 3:n)))
    # mDx = BandedMatrix(view(Dx, :, 3:n), (0, 0))
    mSx = view(mSx, :, 3:n)
    mDx = view(Dx, :, 3:n)
    mul!(F, Dy * transpose(G), transpose(view(Sx, :, 1:2)), -1, true)

    mSy, mDx, mDy, mSx, F, bcy, H, bcx, G
end

function recoverTO(X22::AbstractMatrix, bcy::AbstractMatrix, H::AbstractMatrix, bcx::AbstractMatrix, G::AbstractMatrix)
    # recover full X from boundary conditions
    Ky, Kx = size(H, 1), size(G, 1)
    m, n = size(X22)
    @assert m+Ky == size(G, 2) && n+Kx == size(H, 2) "Incompatible bcs"
    X = Matrix{eltype(X22)}(undef, m+Ky, n+Kx)
    copyto!(view(X, Ky+1:Ky+m, Kx+1:Kx+n), X22)

    # X12 block
    X12 = copyto!(view(X, 1:Ky, Kx+1:Kx+n), view(H, 1:Ky, Kx+1:Kx+n))
    mul!(X12, view(bcy, 1:Ky, Kx+1:Kx+n), X22, -1, true)

    # X11 and X21 block
    X1 = copyto!(view(X, :, 1:Kx), transpose(G))
    mul!(X1, view(X, :, Kx+1:Kx+n), transpose(view(bcx, 1:Kx, Kx+1:Kx+n)), -1, true)

    X
end

# # A function that is duplication of chebop2.bartelsStewart
# function bartelsStewart(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, D::AbstractMatrix, E::AbstractMatrix)
#     #BARTELSSTEWART   Solution to generalized Sylvester matrix equation. 
#     # 
#     # Computes the solution to the Sylvester equation
#     #
#     #         AXB^T + CXD^T = E
#     #
#     # by using the Bartels--Stewart algorithm, see 
#     #
#     # J. D. Gardiner, A. J. Laub, J. J. Amato, & C. B. Moler, Solution of 
#     # the Sylvester matrix equation AXB^T+ CXD^T= E, ACM Transactions on 
#     # Mathematical Software (TOMS), 18(2), 223-231.
#     # 
#     # This Bartels--Stewart solver also takes information xsplit, ysplit so
#     # that if possible it decouples the even and odd modes.
    
#     # Copyright 2017 by The University of Oxford and The Chebfun2 Developers.
#     # See http://www.chebfun.org/ for Chebfun information.

    
#     # factorization in y-direction
#     schury = schur(A, C)
#     P, S = schury.S, schury.T
#     Q1, Z1 = schury.Q', schury.Z

#     # factorization in x-direction
#     schurx = schur(D, B)
#     T, R = schurx.S, schurx.T
#     Q2, Z2 = schurx.Q', schurx.Z
    
#     # Now use the generalised Bartels--Stewart solver found in Gardiner et al.
#     # (1992).  The Sylvester matrix equation now contains quasi upper-triangular
#     # matrices and we can do a backwards substitution.
    
#     # transform the righthand side.
#     F = Q1*E*transpose(Q2)
    
#     # Solution will be a m by n matrix.
#     m = size(A, 1)
#     n = size(B, 1)
#     Y = zeros(eltype(A), m, n)
    
#     # Do a backwards substitution type algorithm to construct the solution.
#     k=n
#     Yfactor = zeros(eltype(A), m, m)
    
#     # Construct columns n,n-1,...,3,2 of the transformed solution.  The first
#     # column is treated as special at the end.
#     while k > 1
#         # There are two cases, either the subdiagonal contains a zero
#         # T(k,k-1)=0 and then it is a backwards substitution, or T(k,k-1)~=0
#         # and then we solve a 2x2 system instead.
        
#         if T[k,k-1] == 0 
#             # Simple case (almost always end up here).
#             rhs = view(F, :, k)
#             if k < n
#                 for jj = k+1:n
#                     yj = view(Y, :, jj)
#                     mul!(rhs, P, yj, -R[k, jj], true)
#                     mul!(rhs, S, yj, -T[k, jj], true)
#                 end
#                 # PY(:,k+1) = P*Y(:,k+1);
#                 # SY(:,k+1) = S*Y(:,k+1);
                
#                 # for jj = k+1:n
#                 #     rhs = rhs - R(k,jj)*PY(:,jj) - T(k,jj)*SY(:,jj);
#                 # end
#             end
            
#             # find the kth column of the transformed solution.
#             copyto!(Yfactor, P)
#             axpby!(T[k, k], S, R[k, k], Yfactor)
#             ldiv!(view(Y, :, k), lu!(Yfactor), rhs)

#             # Y(:,k) = (R(k,k)*P + T(k,k)*S) \ rhs;
            
#             # go to next column
#             k = k-1
#         else
#             # This is a straight copy from the Gardiner et al. paper, and just
#             # solves for two columns at once. (works because of
#             # quasi-triangular matrices.
            
#             # Operator reduction.
#             rhs1 = view(F, :, k-1)
#             rhs2 = view(F, :, k)
            
#             for jj = k+1:n
#                 yj = view(Y, :, jj)
#                 mul!(rhs1, P, yj, -R[k-1, jj], true)
#                 mul!(rhs1, S, yj, -T[k-1, jj], true)
#                 mul!(rhs2, P, yj, -R[k, jj], true)
#                 mul!(rhs2, S, yj, -T[k, jj], true)
#                 # rhs1 = rhs1 - R(k-1,jj)*P*yj - T(k-1,jj)*S*yj;
#                 # rhs2 = rhs2 - R(k,jj)*P*yj - T(k,jj)*S*yj;
#             end
            
#             # 2 by 2 system.
#             SM = zeros(eltype(A), 2*n, 2*n)
#             up = 1:n
#             down = n+1:2*n
            
#             SM[up,up] = R[k-1,k-1]*P + T[k-1,k-1]*S
#             SM[up,down] = R[k-1,k]*P + T[k-1,k]*S
#             SM[down,up] = R[k,k-1]*P + T[k,k-1]*S
#             SM[down,down] = R[k,k]*P + T[k,k]*S
    
#             # Solve
#             UM = SM \ [rhs1; rhs2]
            
#             Y[:,k-1] = view(UM, up) 
#             Y[:,k] = view(UM, down)
            
#             # We solved for two columns so go two columns further.
#             k=k-2
#         end
        
#     end
    
#     if ( k == 1 )
#         # Now we have just the first column to compute.
#         rhs = view(F, :, 1)
#         for jj = 2:n
#             yj = view(Y, :, jj)
#             mul!(rhs, P, yj, -R[1, jj], true)
#             mul!(rhs, S, yj, -T[1, jj], true)
#             # rhs = rhs - R(1,jj)*PY(:,jj) - T(1,jj)*SY(:,jj);
#         end
#         copyto!(Yfactor, P)
#         axpby!(T[1, 1], S, R[1, 1], Yfactor)
#         ldiv!(view(Y, :, 1), lu!(Yfactor), rhs)
#         # Y(:,1) = (R(1,1)*P + T(1,1)*S) \ rhs;
#     end
    
#     # We have now computed the transformed solution so we just transform it
#     # back.
#     X = Z1*Y*transpose(Z2)
# end