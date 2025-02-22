# functions needed for transforming Chebyshev basis to ultraspherical basis and Legenre basis
# only for algorithms in D. Fortunato and A. Townsend, Fast Poisson solvers for spectral methods, IMA Journal of Numerical Analysis, 40 (2019), pp. 1994–2018.
# These functions are duplications of functions in https://github.com/danfortunato/fast-poisson-solvers/tree/master/code/transforms
using FastTransforms

function cheb2ultra( X )
    # CHEB2ULTRA   Convert vector of Chebyshev coefficients to C^(3/2).
    #
    #    CHEB2ULTRA(X) applies the conversion to each column of X if X is a
    #    matrix.
    
    # First convert the matrix of Chebyshev coefficients to a matrix of
    # Legendre coefficients:
    m = size( X, 1 )
    if ( m <= 10000 ) # Determined experimentally
        S = cheb2leg_mat( m )
        X = S * X
    else
        X = FastTransforms.cheb2leg( X )
    end
    
    # Now, convert the matrix of Legendre coefficients to a matrix of
    # ultraspherical coefficients:
    S = leg2ultra_mat( m )
    X = S * X
    
    X
end

function ultra1mx2cheb( X )
    # ULTRA1MX2CHEB    Convert vector of (1-x^2)C^(3/2) coefficients to
    # Chebyshev.
    #
    #  ULTRA1MX2CHEB(X) applies the conversion each column of X if X is
    #    matrix.
    
    # First, convert the matrix of (1-x^2)C^(3/2)(x) coefficients
    # to Legendre coefficients:
    m = size( X, 1 )
    
    S = ultra1mx2leg_mat( m )
    X = S * X
    
    # Now, convert the matrix of Legendre coefficient to a matrix of Chebyshev
    # coefficients:
    if ( m <= 10000 ) # Determined experimentally
        S = leg2cheb_mat( m )
        X = S * X
    else
        X = FastTransforms.leg2cheb( X )
    end
    
    X
end
    
function leg2ultra_mat( n )
    # LEG2ULTRA_MAT Conversion matrix from Legendre coefficients to C^(3/2).
    #
    # Given coefficients in the Legendre basis the C^(3/2) coefficients
    # can be computed via
    #
    #     c = rand(10, 1);    # Legendre coefficients
    #     S = leg2ultra_mat( length(c) ); # conversion matrix
    #     d = S * c;           # C^(3/2) coefficients
    
    # Alex Townsend, 5th May 2016
    
    lam = 1/2
    dg = lam./(lam .+ (2:n-1))
    v  = [1 ; lam./(lam+1) ; dg]
    w  = -dg
    S  = spdiagm(n, n, 0=>v, 2=>w)
    
    S
end
    
function ultra1mx2leg_mat( n )
    # ULTRA1MX2LEG_MAT Conversion matrix for (1-x^2)C^(3/2) to Legendre.
    #
    # Given coefficients in the (1-x^2)C^(3/2) basis the Legendre coefficients
    # can be computed via
    #
    #     c = rand(10, 1);     # (1-x^2)C^(3/2) coefficients
    #     S = ultra1mx2leg_mat( length(c) ); # conversion matrix
    #     c_leg = S * c;       # Legendre coefficients
    #
    
    # Alex Townsend, 5th May 2016
    
    S = spdiagm(0 => (1:n).*(2:(n+1))./2 ./ (3/2:n+1/2))
    S = spdiagm(0=>Fill(1, n), -2=>Fill(-1, n-2)) * S
    
    S
end
    
function cheb2leg_mat( N )
    # CHEB2LEG_MAT Construct the cheb2leg conversion matrix.
    
    # This for-loop is a faster and more accurate way of doing:
    # Lambda = @(z) exp(gammaln(z+1/2) - gammaln(z+1));
    # vals = Lambda( (0:2*N-1)'/2 );
    vals = zeros(2*N,1)
    vals[1] = sqrt(pi)
    vals[2] = 2/vals[1]
    for i = 2:2:2*(N-1)
        vals[i+1] = vals[i-1]*(1-1/i)
        vals[i+2] = vals[i]*(1-1/(i+1))
    end
    
    L = zeros(N, N)
    for j = 0:N-1
        for k = j+2:2:N-1
            L[j+1, k+1] = -k*(j+.5)*(vals[(k-j-2)+1]./(k-j)).*(vals[(k+j-1)+1]./(j+k+1))
        end
    end
    c = sqrt(pi)/2
    for j = 1:N-1
        L[j+1, j+1] = c./vals[ 2*j+1 ]
    end
    L[1,1] = 1   
    
    L
end
    
function leg2cheb_mat( N )
    # LEG2CHEB_MAT Construct the leg2cheb conversion matrix.
    
    # This for-loop is a faster and more accurate way of doing:
    # Lambda = @(z) exp(gammaln(z+1/2) - gammaln(z+1));
    # vals = Lambda( (0:2*N-1)'/2 );
    vals = zeros(2*N,1)
    vals[1] = sqrt(pi)
    vals[2] = 2/vals[1]
    for i = 2:2:2*(N-1)
        vals[i+1] = vals[i-1]*(1-1/i)
        vals[i+2] = vals[i]*(1-1/(i+1))
    end
    
    M = zeros(N, N)
    for j = 0:N-1
        for k = j:2:N-1
            M[j+1, k+1] = 2/pi*vals[(k-j)+1].*vals[(k+j)+1]
        end
    end
    M[1,:] = .5*M[1,:]
    
    M
end