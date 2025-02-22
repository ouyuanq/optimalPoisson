# basic operators for ultraspherical spectral method and boundary treatment
using SparseArrays, FillArrays
using BandedMatrices:BandedMatrices, inbands_getindex, inbands_setindex!

# differential operator
function diffmat(::Type{T}, m::Integer, n::Integer, lambda::Integer) where T
    # sparse differential matrix
    if lambda == 0
        D = BandedMatrix{T}(Eye(m, n))
    else
        D = BandedMatrix{T}((lambda => 2^(lambda - 1)*factorial(lambda - 1) .* (lambda:min(m+lambda-1, n-1)),), (m, n), (-lambda, lambda))
    end

    D
end

diffmat(n::Integer, lambda::Integer) = diffmat(Float64, n, n, lambda)  # default type
diffmat(m::Integer, n::Integer, lambda::Integer) = diffmat(Float64, m, n, lambda)

# conversion operator
function convertmat(::Type{T}, m::Integer, n::Integer, lambda::Integer, mu::Integer) where T
    # sparse conversion matrix to transfer a C^{λ} series to a C^{μ} series
    # S = S_{mu-1}S_{mu-2}...S_{lambda}, where S_{mu} = (diagonal([1 1 1 ...], 0) + diagonal([-1 -1 -1 ...], 2)) * diagonal([mu/mu mu/(mu+1) mu/(mu+2) ...], 0)
    if lambda < mu
        # S_{mu-1} with total bandwidth 2*(mu-lambda)
        Su = 2*(mu-lambda)
        S = BandedMatrix{T}((0 => Fill(1, n), 2 => Fill(-1, n-2)), (m, n), (0, Su))
        if mu == 1
            for l = Su-1:2:Su+1
                Sl = view(S.data, l, 2:n)
                broadcast!(/, Sl, Sl, 2)
            end
        else
            for l = Su-1:2:Su+1
                Sl = view(S.data, l, :)
                lmul!(mu-1, Sl)
                broadcast!(/, Sl, Sl, mu-1:mu+n-2)
            end
        end
        Sdata = S.data

        # multiplication from left to right
        @inbounds for k = mu-2:-1:lambda
            k2 = 2*(mu-k-1)  # bandwidth before multiplication
            for j = n:-1:3
                j2 = j-2
                for i = j2:-2:j2-k2
                    Sij = inbands_getindex(Sdata, Su, i, j) - inbands_getindex(Sdata, Su, i, j2)
                    inbands_setindex!(Sdata, Su, Sij, i, j)
                end
            end
            # scaling matrix
            if k == 0
                for l = Su-k2-1:2:Su+1
                    Sl = view(S.data, l, 2:n)
                    broadcast!(/, Sl, Sl, 2)
                end
            else
                for l = Su-k2-1:2:Su+1
                    Sl = view(S.data, l, :)
                    lmul!(k, Sl)
                    broadcast!(/, Sl, Sl, k:k+n-1)
                end
            end
        end
    elseif lambda == mu
        S = BandedMatrix{T}(Eye(m, n), (0, 0))
    else
        @error "μ should be no less than λ"
    end
    
    S
end

convertmat(n::Integer, lambda::Integer, mu::Integer) = convertmat(Float64, n, n, lambda, mu)  # default type
convertmat(m::Integer, n::Integer, lambda::Integer, mu::Integer) = convertmat(Float64, m, n, lambda, mu)

# multiplication operator
function multmatT(n::Integer, a::AbstractVector{T}) where T
    # construct a dense multiplication matrix which represents multiplying a Chebyshev basis coefficients u to a Chebyshev basis coefficients. The resulting maxtrix has dimension n × n

    la = length(a)
    # @assert la <= n "Too small dimension of multiplication matrix"
    # empty term
    if la == 0
        M = BandedMatrix{T}((0 => Fill(0, n)), (n, n), (0, 0))
        return M
    end
    if la == 1
        # Multiplying by a scalar is easy
        M = BandedMatrix{T}((0 => Fill(a[1], n)), (n, n), (0, 0))
        return M
    end

    M = BandedMatrix{T}(undef, n, n, la-1, la-1)
    M[band(0)] .= a[1]
    # Toeplitz part
    @inbounds for i = 2:la
        M[band(i-1)] .= a[i] / 2
        M[band(1-i)] .= a[i] / 2
    end
    # Hankel part
    @inbounds for j = 1:la-1
        for i = 2:la-j+1
            M[i, j] += a[i+j-1] / 2
        end
    end

    M
end

# transformation operator
# second order Dirichlet
function DDtransform(::Type{T}, m::Integer, n::Integer) where T
    # transformation operator related to left and right Dirichlet conditions
    R = BandedMatrix{T}((0=>Fill(1, min(m, n)), -2 => Fill(-1, min(m-2, n))), (m, n), (2, 0))
    # rdiv!(R.data, Diagonal(4:2:2*n+2))

    R
end

# second order left Dirichlet and right Neumann
function DNtransform(::Type{T}, m::Integer, n::Integer) where T
    # transformation operator related to left Dirichlet and right Neumann conditions
    @assert m > n "Wrong dimension for transformation operator"
    
    den = (1:m+1).^2
    axpy!(true, view(den, m-1:-1:1), view(den, m:-1:2))

    R = BandedMatrix{T}((0 => -view(den, 2:min(m, n)+1), -1 => -(T(4):4:4*min(m-1, n)) , -2 => view(den, 1:min(m-2, n))), (m, n), (2, 0))
    rdiv!(R.data, Diagonal(view(den, 1:n)))
    rdiv!(R.data, Diagonal(4:2:2n+2))  # for numerical stability
    # R = spdiagm(m, n, 0 => -view(den, 2:min(m, n)+1), -1 => -(T(4):4:4*min(m-1, n)) , -2 => view(den, 1:min(m-2, n)))
    # rdiv!(R, Diagonal(view(den, 1:n)))
    # rdiv!(R, Diagonal(4:2:2n+2))  # for numerical stability

    R
end

# second order left Dirichlet and right Robin
function DRtransform(::Type{T}, m::Integer, n::Integer, alpha::Number) where T
    # transformation operator related to left Dirichlet and right Robin conditions
    @assert m > n "Wrong dimension for transformation operator"

    den = (1:m+1).^2
    axpy!(true, view(den, m-1:-1:1), view(den, m:-1:2))
    lmul!(alpha, den)
    broadcast!(+, den, den, 2)

    R = BandedMatrix{T}((0 => -view(den, 2:min(m, n)+1), -1 => -alpha.*(T(4):4:4*min(m-1, n)) , -2 => view(den, 1:min(m-2, n))), (m, n), (2, 0))
    rdiv!(R.data, Diagonal(view(den, 1:n)))
    rdiv!(R.data, Diagonal(4:2:2n+2))  # for numerical stability
    # R = spdiagm(m, n, 0 => -view(den, 2:min(m, n)+1), -1 => -alpha.*(T(4):4:4*min(m-1, n)) , -2 => view(den, 1:min(m-2, n)))
    # rdiv!(R, Diagonal(view(den, 1:n)))
    # rdiv!(R, Diagonal(4:2:2n+2))  # for numerical stability

    R
end

# fourth order Dirichlet and Neumann
function DNDNtransform(::Type{T}, m::Integer, n::Integer) where T
    # transformation operator related to Dirichlet and Neumann conditions
    @assert m > n "Wrong dimension for transformation operator"

    R = BandedMatrix{T}((0 => 3:n+2, -2 => -(4:2:2n+2), -4 => 1:n), (m, n), (4, 0))
    # rdiv!(R.data, Diagonal(48 .* (4:n+3)))
    # R = spdiagm(m, n, 0 => T(3):n+2, -2 => -(4:2:2n+2), -4 => 1:n)
    # rdiv!(R.data, Diagonal(48 .* (4:n+3)))

    R
end

## operator in D. Fortunato and A. Townsend, Fast Poisson solvers for spectral methods, IMA Journal of Numerical Analysis, 40 (2019), pp. 1994–2018
function diffmultmat(::Type{T}, n::Integer) where T
    # pentadiagonal matrix for (1-x^2)*(d^2/(dx)^2)
    jj = 0:n-1
    dsub = -1 ./ (2*(jj.+3/2)) .* (jj.+1) .* (jj.+2) .* (1/2 ./(1/2 .+ jj.+2))
    dsup = -1 ./ (2*(jj.+3/2)) .* (jj.+1) .* (jj.+2) .* (1/2 ./(1/2 .+ jj))
    d = -dsub - dsup
    Mn = BandedMatrix{T}((-2 => view(dsub, 1:n-2), 0 => d, 2 => view(dsup, 3:n)), (n, n))
    # Construct D^{-1}, which undoes the scaling from the Laplacian identity
    # invDn = Diagonal{T}(-1 ./ (jj.*(jj.+3).+2))
    # Tn = invDn * Mn  # a different way
    invDn = lmul!(-1, Diagonal{T}(jj.*(jj.+3).+2))
    Tn = ldiv!(invDn, Mn)  # only ldiv! is supported for BandedMatrix

    Tn, invDn
end