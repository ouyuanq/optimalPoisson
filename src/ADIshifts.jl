# construct and reorder shifts for ADI iterations
using Elliptic: Jacobi, F as ellipticF, K as ellipke

# These functions are duplications of functions in https://github.com/danfortunato/fast-poisson-solvers/blob/master/code/adi/ADIshifts.m
function ADIshifts(a::Number, b::Number, c::Number, d::Number, tol::Number)
    #ADISHIFTS  ADI shifts for solving AX-XB=F.
    # [P, Q] = ADISHIFTS(a, b, c, d, TOL) returns the optimal ADI shifts for
    # solving AX-XB = F with the ADI method when A and B have eigenvalues lying
    # in [a,b] and [c,d], respectively. WLOG, we require that a<b<c<d and
    # 0<tol<1.

    gam = (c - a) * (d - b) / (c - b) / (d - a)               # Cross-ratio of a,b,c,d
    # Calculate Mobius transform T:{-alp,-1,1,alp}->{a,b,c,d} for some alp:
    alp = -1 + 2 * gam + 2 * sqrt(gam^2 - gam)        # Mobius exists with this t
    A = a * (alp * (c - b) + (b + c)) - 2 * b * c                  # Determinant formulae for Mobius
    B = a * (alp * (b + c) + (c - b)) - 2 * alp * b * c
    C = alp * (c - b) + (2 * a - b - c)
    D = (c - b) + alp * (2 * a - b - c)
    T = z -> (A .* z + B) ./ (C .* z + D)                   # Mobius transfom
    J = ceil(log(16 * gam) * log(4 / tol) / pi^2)     # Number of ADI iterations
    if alp > 1e7
        K = (2 * log(2) + log(alp)) + (-1 + 2 * log(2) + log(alp)) / alp^2 / 4
        m1 = 1 / alp^2
        u = (1/2:J-1/2) * K / J
        dn = sech.(u) + 0.25 * m1 * (sinh.(u) .* cosh.(u) + u) .* tanh.(u) .* sech.(u)
    else
        K = ellipke(1 - 1 / alp^2)
        dn = Jacobi.dn.((1/2:J-1/2) .* (K / J), 1 - 1 / alp^2) # ADI shifts for [-1,-1/t]&[1/t,1]
    end

    p, q = alp * dn, -alp * dn
    map!(T, p, p), map!(T, q, q)
end

function ADIshifts(a::Number, b::Number, tol::Number)
    # convenient way for the same operator in x- and y-direction
    @assert a * b > 0 "Intersecting spectra"

    if a < 0
        ma, mb = min(a, b), max(a, b)
        p, q = ADIshifts(ma, mb, -mb, -ma, tol)
    else
        ma, mb = min(a, b), max(a, b)
        p, q = ADIshifts(-mb, -ma, ma, mb, tol)
    end

    p, q
end