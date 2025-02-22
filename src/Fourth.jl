# set up matrices for solving fourth-order equation u_{xxxx} + u_{yyyy} = f

function fourth_init(::Type{T}, m::Integer, n::Integer, f::Function, lbc, rbc, dbc, ubc) where T
    # initialization for solving fourth-order Poisson-like equation u_xxxx + u_yyyy = f with Dirichlet and Neumann boundary conditions
    
    # set up the operators
    Sm = convertmat(T, m-4, m, 0, 4)
    Dm = diffmat(T, m-4, m, 4)
    Tm = DNDNtransform(T, m, m-4)
    STm, DTm = Sm * Tm, Dm * Tm
    if n == m
        # save for the same dimension of x- and y-direction
        Sn, Dn, Tn, STn, DTn = Sm, Dm, Tm, STm, DTm
    else
        Sn = convertmat(T, n-4, n, 0, 4)
        Dn = diffmat(T, n-4, n, 4)
        Tn = DNDNtransform(T, n, n-4)
        STn, DTn = Sn * Tn, Dn * Tn
    end

    # rhs
    F = Sm * coeffs2(f, m-1, n-1, T) * transpose(Sn)

    # interpolation for BCs
    g = interpDNDN(coeffs.(lbc, T), coeffs.(rbc, T), coeffs.(dbc, T), coeffs.(ubc, T), m, n)

    # modified rhs (F = F - g_xxxx - g_yyyy)
    axpy!(-1, Sm * g * transpose(Dn), F)
    axpy!(-1, Dm * g * transpose(Sn), F)

    STm, DTn, DTm, STn, F, Tm, Tn, g
end

function lowerboundfourthdiffDNDN(n::Integer)
    # lower bound for eigenvalues of fourth-order differential equations with Dirichlet conditions
    n2 = Float64(n-1)
    (sqrt(3628800(n2-1)*(n2-2))*n2*(n2+1)*(n2+2)) / sqrt(3712n2^8 + 11136n2^7 - 11136n2^6 - 55680n2^5 - 22272n2^4 + 44544n2^3 + 29696n2^2 - 70875n2 + 28350)
end

function upperboundfourthdiffDNDN(n::Integer)
    # upper bound for eigenvalues of fourth-order differential equations with Dirichlet conditions
    # note that this is first order of root bound of Newton since largest eigenvalues are complex pair
    n2 = Float64(n-1)
    ub = (2(n2-3)*(n2-2)*(n2+8)*(n2+9)/109395)*(n2^4 + 12n2^3 + 281n2^2 + 1470n2 - 6300)
    ub / 2  # for complex conjugate pair
end