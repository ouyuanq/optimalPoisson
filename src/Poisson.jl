# set up Poisson equations

function poisson_init(::Type{T}, m::Integer, n::Integer, f::Function, lbc::Function, rbc::Function, dbc::Function, ubc::Function) where T
    # initialization for solving Poisson equation u_xx + u_yy = f with Dirichlet boundary conditions
    
    # set up the operators
    Sm = convertmat(T, m-2, m, 0, 2)
    Dm = diffmat(T, m-2, m, 2)
    Tm = DDtransform(T, m, m-2)
    STm, DTm = Sm * Tm, Dm * Tm
    if n == m
        # save for the same dimension of x- and y-direction
        Sn, Dn, Tn, STn, DTn = Sm, Dm, Tm, STm, DTm
    else
        Sn = convertmat(T, n-2, n, 0, 2)
        Dn = diffmat(T, n-2, n, 2)
        Tn = DDtransform(T, n, n-2)
        STn, DTn = Sn * Tn, Dn * Tn
    end

    # rhs
    F = Sm * coeffs2(f, m-1, n-1, T) * transpose(Sn)

    # interpolation for BCs
    g = interpDD(coeffs(lbc, T), coeffs(rbc, T), coeffs(dbc, T), coeffs(ubc, T), m, n)

    # modified rhs (F = F - g_xx - g_yy)
    mul!(F, Sm, g * transpose(Dn), -1, true)
    mul!(F, Dm, g * transpose(Sn), -1, true)

    STm, DTn, DTm, STn, F, Tm, Tn, g
end

function lowerboundsecdiffDD(n::Integer)
    # lower bound for eigenvalues of second-order differential equations with Dirichlet conditions
    n2 = Float64(n-1)
    lb = ((n2-1)*n2*(n2+4)*(n2+5)/121275)*(29*n2^4+232*n2^3+2279*n2^2+7260*n2-17640)
    -sqrt(lb)
end

function upperboundsecdiffDD(n::Integer)
    # upper bound for eigenvalues of second-order differential equations with Dirichlet conditions
    -pi^2 / 4
end


# set up for D. Fortunato and A. Townsend, Fast Poisson solvers for spectral methods, IMA Journal of Numerical Analysis, 40 (2019), pp. 1994–2018.
function poissonFT_init(::Type{T}, m::Integer, n::Integer, f::Function, lbc::Function, rbc::Function, dbc::Function, ubc::Function) where T
    # initialization for solving Poisson equation u_xx + u_yy = f with Dirichlet boundary conditions
    
    # set up the operators
    Tm, invDm = diffmultmat(T, m)
    Sm = convertmat(T, m, m, 0, 2)
    Dm = diffmat(T, m, m, 2)
    if n == m
        Tn, invDn = Tm, invDm
        Sn, Dn = Sm, Dm
    else
        Tn = diffmultmat(T, n)
        Sn = convertmat(T, n, n, 0, 2)
        Dn = diffmat(T, n, n, 2)
    end

    # interpolation for BCs
    g = interpDD(coeffs(lbc, T), coeffs(rbc, T), coeffs(dbc, T), coeffs(ubc, T), m, n)

    # modified rhs (F = F - g_xx - g_yy)
    F = coeffs2(f, m-1, n-1, T)
    axpy!(-1, Sm \ (Dm * g), F)
    axpy!(-1, transpose(Sn \ (Dn * transpose(g))), F)

    # transform to C^{3/2} basis
    F = cheb2ultra(transpose(cheb2ultra(transpose(F))))
    ldiv!(invDm, rdiv!(F, invDn))
    # lmul!(invDm, rmul!(F, invDn))  # in case invDm is constructed in a different way

    Tm, Tn, F, g
end

function poissonOT_init(::Type{T}, m::Integer, n::Integer, f::Function, lbc::Function, rbc::Function, dbc::Function, ubc::Function) where T
    # initialization for solving Poisson equation u_xx + u_yy = f with Dirichlet boundary conditions using ultraspherical spectral with boundary bordering
    
    # set up the operators
    Sm = convertmat(T, m-2, m, 0, 2)
    Dm = diffmat(T, m-2, m, 2)
    lband, uband = max(Sm.l+2, Dm.l+2), max(Sm.u-2, Dm.u-2)
    bcm = ones(T, 2, m)  # Dirichlet BCs
    bcm[1, 2:2:end] .= -1
    # AlmostBandedMatrix
    BSm = BandedMatrix{T}(undef, (m, m), (lband, uband))
    copyto!(view(BSm, 3:m, :), Sm)
    BSm[1, rowrange(BSm, 1)] = -view(bcm, 1, rowrange(BSm, 1))
    BSm[2, rowrange(BSm, 2)] = -view(bcm, 2, rowrange(BSm, 2))
    ABSm = AlmostBandedMatrix(BSm, ApplyMatrix(*, Matrix{T}(I, m, 2), -bcm))
    BDm = BandedMatrix{T}(undef, (m, m), (lband, uband))
    copyto!(view(BDm, 3:m, :), Dm)
    BDm[1, rowrange(BDm, 1)] = view(bcm, 1, rowrange(BDm, 1))
    BDm[2, rowrange(BDm, 2)] = view(bcm, 2, rowrange(BDm, 2))
    ABDm = AlmostBandedMatrix(BDm, ApplyMatrix(*, Matrix{T}(I, m, 2), bcm))
    if n == m
        # save for the same dimension of x- and y-direction
        ABSn, ABDn = ABSm, ABDm
        Sn, Dn = Sm, Dm
    else
        Sn = convertmat(T, n-2, n, 0, 2)
        Dn = diffmat(T, n-2, n, 2)
        bcn = ones(T, 2, n)  # Dirichlet BCs
        bcn[1, 2:2:end] .= -1
        # AlmostBandedMatrix
        BSn = BandedMatrix{T}(undef, (n, n), (lband, uband))
        copyto!(view(BSn, 3:n, :), Sn)
        BSn[1, rowrange(BSn, 1)] = -view(bcn, 1, rowrange(BSn, 1))
        BSn[2, rowrange(BSn, 2)] = -view(bcn, 2, rowrange(BSn, 2))
        ABSn = AlmostBandedMatrix(BSn, ApplyMatrix(*, Matrix{T}(I, n, 2), -bcn))
        BDn = BandedMatrix{T}(undef, (n, n), (lband, uband))
        copyto!(view(BDn, 3:n, :), Dn)
        BDn[1, rowrange(BDn, 1)] = view(bcn, 1, rowrange(BDn, 1))
        BDn[2, rowrange(BDn, 2)] = view(bcn, 2, rowrange(BDn, 2))
        ABDn = AlmostBandedMatrix(BDn, ApplyMatrix(*, Matrix{T}(I, n, 2), bcn))
    end

    # interpolation for BCs
    g = interpDD(coeffs(lbc, T), coeffs(rbc, T), coeffs(dbc, T), coeffs(ubc, T), m, n)
    # rhs
    F = axpy!(true, almostbandedlrmul(ABSm, g, ABDn), almostbandedlrmul(ABDm, g, ABSn))
    F[3:m, 3:n] = Sm * coeffs2(f, m-1, n-1, T) * transpose(Sn)

    ABSm, ABDn, ABDm, ABSn, F
end

function poisson_mixedbc_init(::Type{T}, m::Integer, n::Integer, f::Function, lbc::Function, rbc::Function, dbc::Function, ubc::Function, theta::Number) where T
    # initialization for solving Poisson equation u_xx + u_yy = f with left Dirichlet, right Neumann, down Dirichlet and upper Robin conditions
    
    # set up the operators
    Sy = convertmat(T, m-2, m, 0, 2)
    Dy = diffmat(T, m-2, m, 2)

    if n == m
        # save for the same dimension of x- and y-direction
        Sx, Dx = Sy, Dy
    else
        Sx = convertmat(T, n-2, n, 0, 2)
        Dx = diffmat(T, n-2, n, 2)
    end

    Ty = DRtransform(T, m, m-2, theta)
    STy, DTy = Sy * Ty, Dy * Ty

    Tx = DNtransform(T, n, n-2)
    STx, DTx = Sx * Tx, Dx * Tx

    # rhs
    F = Sy * coeffs2(f, m-1, n-1, T) * transpose(Sx)

    # interpolation for BCs
    g = interp_mixed(theta, coeffs(lbc, T), coeffs(rbc, T), coeffs(dbc, T), coeffs(ubc, T), m, n)

    # modified rhs (F = F - g_xx - g_yy)
    mul!(F, Sy, g * transpose(Dx), -1, true)
    mul!(F, Dy, g * transpose(Sx), -1, true)

    STy, DTx, DTy, STx, F, Ty, Tx, g
end

function lowerboundsecdiffDN(n::Integer)
    # lower bound for eigenvalues of second-order differential equations with Dirichlet conditions
    n2 = Float64(n-1)
    lb = ((n2-1)*(n2+4)*2/14189175)*(7636*n2^10+114540*n2^9+1241508*n2^8+8712936*n2^7+18460305*n2^6-69348231*n2^5-232635715*n2^4+300782307*n2^3+697081464*n2^2-1031649750*n2+299295000)/(2*n2^2+6*n2-3)^2
    -sqrt(lb)
end

function upperboundsecdiffDN(n::Integer)
    # upper bound for eigenvalues of second-order differential equations with Dirichlet conditions
    -pi^2 / 16
end

function lowerboundsecdiffDR(n::Integer, theta::Number)
    # lower bound for eigenvalues of second-order differential equations with Dirichlet conditions
    n2 = Float64(n-1)
    lb = ((n2-1)*(n2+4)*2/14189175)*(7636*n2^10*theta^2+114540*n2^9*theta^2+1241508*n2^8*theta^2+83160*n2^8*theta+8712936*n2^7*theta^2+997920*n2^7*theta+18460305*n2^6*theta^2+10881900*n2^6*theta+339300*n2^6-69348231*n2^5*theta^2+66502620*n2^5*theta+3053700*n2^5-232635715*n2^4*theta^2+53069220*n2^4*theta+30057300*n2^4+300782307*n2^3*theta^2-584820540*n2^3*theta+134538300*n2^3+697081464*n2^2*theta^2-396241200*n2^2*theta-66830400*n2^2-1031649750*n2*theta^2+2219506920*n2*theta-764688600*n2+299295000*theta^2-1317060000*theta+801122400)/(theta*(2*n2^2+6*n2-3)+10)^2
    -sqrt(lb)
end

function upperboundsecdiffDR(n::Integer, theta::Number)
    # upper bound for eigenvalues of second-order differential equations with Dirichlet conditions
    n2 = Float64(n-1)
    ub = 720*((n2-1)*n2^2*(n2+1)^2)*(theta+2)^2 / (1920*n2^5*theta^2+1536*n2^5*theta+512*n2^5+1920*n2^4*theta^2+1536*n2^4*theta+512*n2^4-1920*n2^3*theta^2-1536*n2^3*theta-512*n2^3-2010*n2^2*theta^2-1896*n2^2*theta-872*n2^2-45*n2*theta^2-180*n2*theta-180*n2-45*theta^2-180*theta-180)
    -sqrt(ub)
end