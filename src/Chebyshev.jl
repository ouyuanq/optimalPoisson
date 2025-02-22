# methods related to Chebyshev polynomials and coefficients
using FillArrays, FFTW
# Contains code that is based in part on Chebfun v5's chebfun/standardChop, 
# chebfun/@chebtech2/vals2coeffs.m and chebfun/@chebtech2/coeffs2vals.m
# which is distributed with the following license:

# Copyright (c) 2015, The Chancellor, Masters and Scholars of the University
# of Oxford, and the Chebfun Developers. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the University of Oxford nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## one dimensional
function vals2coeffs!(values::AbstractVector{T}) where {T}
    # VALS2COEFFS   Convert values at Chebyshev points to Chebyshev coefficients.
    #    C = VALS2COEFFS(V) returns the (N+1)x1 vector of coefficients such that F(x)
    #    = C(1)*T_0(x) + C(2)*T_1(x) + C(N+1)*T_N(x) (where T_k(x) denotes the
    #    k-th 1st-kind Chebyshev polynomial) interpolates the data [V(1) ; ... ;
    #    V(N+1)] at Chebyshev points of the 2nd kind.

    #  Get the length of the input:
    n = length(values)

    #  Trivial case (constant):
    if n <= 1
        return values
    end

    #  Mirror the values (to fake a DCT using an FFT):
    if T <: Real
        tmp = Vector{Complex{T}}(undef, 2 * n - 2)
    else
        tmp = Vector{T}(undef, 2 * n - 2)
    end
    copyto!(view(tmp, 1:n-1), view(values, n:-1:2))
    copyto!(view(tmp, n:2*n-2), view(values, 1:n-1))

    # FFT
    ifft!(tmp, 1)

    # Truncate
    if isreal(values)
        map!(real, values, view(tmp, 1:n))
    else
        copyto!(values, view(tmp, 1:n))
    end

    #  Scale the interior coefficients:
    lmul!(2, view(values, 2:n-1))

    values
end

function coeffs(f::Function, n::Integer, T=Float64)
    # the input f is a function handle, n is the number of Chebyshev points
    # which is enough for computing the Chebyshev coefficients of f

    x = (sin.(pi * (-T(n):2:n) / (2 * n)))
    map!(f, x, x)
    vals2coeffs!(x)
end

function coeffs(f::Function, T = Float64)
    # find the Chebyshev coefficients for function f which are resolved to machine epsilon
    fc = Vector{T}(undef, 17)
    for i = 4:20
        n = 2^i
        map!(sin, fc, pi * (-n:2:n) / (2n))
        map!(f, fc, fc)
        vals2coeffs!(fc)
        m = standardChop(fc)
        if m < n
            deleteat!(fc, m+1:n+1)
            break
        end

        if i == 20
            error("f is unresolved")
        end

        Base._growend!(fc, n)  # prolong fc
    end

    fc
end

function standardChop(coeffs::AbstractVector{T}, tol = eps(real(T))) where T
    #  Reduce the number of coefficients by dropping the tail that can be discarded.
    #  See J. L. Aurentz and L. N. Trefethen, "Chopping as
    #  Chebyshev series", http://arxiv.org/abs/1512.01803, December 2015.

    #  Check magnitude of TOL:
    if tol >= 1
        return 1
    end

    #  Make sure COEFFS has length at least 16:
    n = length(coeffs)
    cutoff = n
    if  n < 17
        # resort to naive chop
        mx = maximum(abs, coeffs)
        if mx == 0
            return 0
        end
        for k=n:-1:1
            if abs(coeffs[k]) > tol*mx
                return k
            end
        end
        return 0
    end

    #  Step 1: Convert COEFFS to a new monotonically nonincreasing
    #          vector ENVELOPE normalized to begin with the value 1.

    envelope = reverse(abs.(coeffs))
    accumulate!(max, envelope, envelope)
    reverse!(envelope)
    if envelope[1] == 0
        cutoff = 1
        return cutoff
    else
        envelope = envelope ./ envelope[1]
    end

    #  Step 2: Scan ENVELOPE for a value PLATEAUPOINT, the first point J-1, if any,
    #  that is followed by a plateau.  A plateau is a stretch of coefficients
    #  ENVELOPE(J),...,ENVELOPE(J2), J2 = round(1.25*J+5) <= N, with the property
    #  that ENVELOPE(J2)/ENVELOPE(J) > R.  The number R ranges from R = 0 if
    #  ENVELOPE(J) = TOL up to R = 1 if ENVELOPE(J) = TOL^(2/3).  Thus a potential
    #  plateau whose starting value is ENVELOPE(J) ~ TOL^(2/3) has to be perfectly
    #  flat to count, whereas with ENVELOPE(J) ~ TOL it doesn't have to be flat at
    #  all.  If a plateau point is found, then we know we are going to chop the
    #  vector, but the precise chopping point CUTOFF still remains to be determined
    #  in Step 3.

    plateauPoint = 0
    for j = 2:n
        j2 = round(Int, 1.25*j + 5)
        if j2 > n
            #  there is no plateau: exit
            return cutoff
        end
        e1 = envelope[j]
        e2 = envelope[j2]
        r = 3*(1 - log(e1)/log(tol))
        plateau = (e1 == 0) | (e2/e1 > r)
        if plateau
            #  a plateau has been found: go to Step 3
            plateauPoint = j - 1
            break
        end
    end

    #  Step 3: fix CUTOFF at a point where ENVELOPE, plus a linear function
    #  included to bias the result towards the left end, is minimal.
    #
    #  Some explanation is needed here.  One might imagine that if a plateau is
    #  found, then one should simply set CUTOFF = PLATEAUPOINT and be done, without
    #  the need for a Step 3. However, sometimes CUTOFF should be smaller or larger
    #  than PLATEAUPOINT, and that is what Step 3 achieves.
    #
    #  CUTOFF should be smaller than PLATEAUPOINT if the last few coefficients made
    #  negligible improvement but just managed to bring the vector ENVELOPE below the
    #  level TOL^(2/3), above which no plateau will ever be detected.  This part of
    #  the code is important for avoiding situations where a coefficient vector is
    #  chopped at a point that looks "obviously wrong" with PLOTCOEFFS.
    #
    #  CUTOFF should be larger than PLATEAUPOINT if, although a plateau has been
    #  found, one can nevertheless reduce the amplitude of the coefficients a good
    #  deal further by taking more of them.  This will happen most often when a
    #  plateau is detected at an amplitude close to TOL, because in this case, the
    #  "plateau" need not be very flat.  This part of the code is important to
    #  getting an extra digit or two beyond the minimal prescribed accuracy when it
    #  is easy to do so.

    if plateauPoint != 0 && envelope[plateauPoint] == 0
        cutoff = plateauPoint
    else
        j3 = sum(envelope .>= tol^(7/6))
        if j3 < j2
            j2 = j3 + 1
            envelope[j2] = tol^(7/6)
        end
        @views cc = log10.(envelope[1:j2])
        axpy!(true, range(0, (-1/3)*log10(tol), length = j2), cc)
        d = argmin(cc)
        cutoff = max(d - 1, 1)
    end

    return cutoff
end

## two-dimensional
function vals2coeffs2D!(values::AbstractMatrix{T}) where {T}
    # Convert values at 2D Chebyshev points to tensor product of Chebyshev coefficients

    # perform 1D transformation along each dimension
    vals2coeffs!(transpose(values))
    vals2coeffs!(values)
end

function vals2coeffs!(values::AbstractMatrix{T}) where {T}
    # Convert values at Chebyshev points to Chebyshev coefficients along each column of values

    #  Get the length of the coefficients
    n = size(values, 1)

    #  Trivial case (constant):
    if n <= 1
        return
    end

    # Mirror the values (to fake a DCT using an FFT):
    tmp = Matrix{complex(T)}(undef, 2 * n - 2, size(values, 2))
    copyto!(view(tmp, 1:n-1, :), view(values, n:-1:2, :))
    copyto!(view(tmp, n:2*n-2, :), view(values, 1:n-1, :))

    # FFT along each column
    ifft!(tmp, 1)

    # Truncate
    if isreal(values)
        map!(real, values, view(tmp, 1:n, :))
    else
        copyto!(values, view(tmp, 1:n, :))
    end

    #  Scale the interior coefficients:
    lmul!(2, view(values, 2:n-1, :))

    values
end

function coeffs2vals(coeffs::AbstractVector{T}) where T<:Number
    # Convert Chebyshev coefficients to values at Chebyshev points of the 2nd kind.

    #  Get the length of the input:
    n = length(coeffs)

    #  Trivial case (constant or empty):
    if ( n <= 1 )
        return copy(coeffs)
    end

    #  check for symmetry
    isEven = mapreduce(abs, max, view(coeffs, 2:2:n)) == 0
    isOdd = mapreduce(abs, max, view(coeffs, 1:2:n)) == 0

    # Mirror the coefficients (to fake a DCT using an FFT):
    tmp = reverse(coeffs)
    pop!(tmp); popfirst!(tmp)
    prepend!(tmp, coeffs)
    ldiv!(2, tmp)
    tmp[1], tmp[n] = coeffs[1], coeffs[n]

    if isreal(coeffs)
        #  Real-valued case:
        values = real(fft(tmp))
    elseif isreal(1im*coeffs)
        #  Imaginary-valued case:
        values = 1im*real(fft(imag(tmp)))
    else
        #  General case:
        values = fft(tmp)
    end

    #  Flip and truncate:
    deleteat!(values, n+1:2*n-2)
    reverse!(values)

    #  enforce symmetry
    if isEven
        axpy!(true, reverse(values), values)
        ldiv!(2, values)
    end

    if isOdd
        axpy!(-1, reverse(values), values)
        ldiv!(2, values)
    end

    return values
end

function chebpts(n::Integer, T = Float64)
    if n == 0
        return zeros(T, 0)
    elseif n == 1
        return zeros(T, 1)
    else
        m = n - 1
        x = lmul!(pi/2m, Vector{T}(-m:2:m))  # (Use of sine enforces symmetry.)
        map!(sin, x, x)
        return x
    end
end

function coeffs2(f::Function, m::Integer, n::Integer, T=Float64)
    # Compute the tensor product Chebyshev coefficient of 2D function f over a m x n Chebyshev grid

    xcheb = T.(sin.(pi * (-n:2:n) / (2 * n)))
    ycheb = T.(sin.(pi * (-m:2:m) / (2 * m)))
    if applicable(f, 0, 0)
        # f = (x, y) -> ...
        vals = map(x -> f(x[2], x[1]), Base.product(ycheb, xcheb))
    else
        # f = x -> x[1]...x[2]...
        # note that y axis first
        vals = map(x -> f((x[2], x[1])), Base.product(ycheb, xcheb))
    end

    vals2coeffs2D!(vals)
end

# function coeffsaxpy!(alpha::Number, v::AbstractVector{T}, u::AbstractVector{U}) where {T, U}
#     # return u = alpha * v + u, where u and v may be of different lengths but the same type
#     # The result is stored in u

#     lu, lv = length(u), length(v)
#     if lu < lv
#         # prolong the destination
#         append!(u, Zeros(T, lv - lu))
#         axpy!(alpha, v, u)
#     else
#         axpy!(alpha, v, view(u, 1:lv))
#     end

#     cutoff = standardChop(u)
#     deleteat!(u, cutoff+1:length(u))

#     u
# end

function detresolution(X::AbstractMatrix, tol::Number)
    # determine whether the solution X is resolved
    m, n = size(X)
    normX = norm(X)
    
    # y-direction resolution
    yres = (norm(view(X, m-5:m, :)) / normX) < tol

    # x-direction resolution
    xres = (norm(view(X, :, n-5:n)) / normX) < tol

    yres && xres
end