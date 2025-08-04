# solve Poisson equation by finite difference (FD) method
# The underlying matrix equation is AX + XA = F, where A is a tridiagonal Toeplitz matrix -1/h^2(diag(2*ones(n), 0) + diag(ones(n-1), 1) + diag(ones(n-1)), -1).
# The solution can be expressed as X = S(C ∘ (S^{-1} F S^{-T}))S^T, C_{jk} = 1/(λ_j + λ_k), λ_k = -4/h^2 sin^2(πk/(2n)), 1 ≤ k ≤ n-1, and S is the normalized discrete sine transformation (of type I). Note that S = S^T = S^{-1}.

function Poisson_FD(F::AbstractMatrix{T}) where {T}
    n, m = size(F)
    lam_y = -(n+1)^2*(sin.(T(pi)/(2*(n+1)) * (1:n)).^2) # 4 times smaller than eigenvalues on [0, 1] x [0, 1]
    # lam_x = -(m+1)^2*(sin.(T(pi)/(2*(m+1)) * (1:m)).^2)
    fftplan_y = plan_fft!(ones(Complex{T}, 2n+2)) #
    if n == m
        fftplan_x = fftplan_y
    else
        fftplan_x = plan_fft!(ones(Complex{T}, 2m+2))
    end
    X = copy(F)

    double_dst!(X, fftplan_y, fftplan_x)
    lmul!(4/((m+1)*(n+1)), X)
    for j in axes(X, 2)
        lam_xj = -(m+1)^2*(sin(T(pi)/(2*(m+1)) * j).^2)
        axpy!(lam_xj, Ones(T, n), lam_y)
        Xj = view(X, :, j)
        broadcast!(/, Xj, Xj, lam_y)
        axpy!(-lam_xj, Ones(T, n), lam_y)
    end

    double_dst!(X, fftplan_y, fftplan_x)

    X
end

function double_dst!(F::AbstractMatrix{T}, fftplan_y, fftplan_x) where {T}
    n, m = size(F)
    tempy = zeros(Complex{T}, 2n+2)
    
    dst!(F, fftplan_y, tempy)

    if n == m
        dst!(transpose!(F), fftplan_x, tempy)
        transpose!(F)
    else
        tempx = zeros(Complex{T}, 2m+2)
        Ft = Matrix(transpose(F))
        dst!(Ft, fftplan_x, tempx)
        copyto!(F, transpose(Ft))
    end

    F
end

function dst!(F::AbstractMatrix{T}, fftplan, temp::AbstractVector{Complex{T}}) where {T}
    # simulate discrete sine transformation via FFT
    n = size(F, 1)
    n2 = length(temp)
    for j in axes(F, 2)
        temp[1] = 0
        temp[n+2] = 0
        copyto!(view(temp, 2:n+1), view(F, :, j))
        lmul!(-1, copyto!(view(temp, n+3:n2), view(F, n:-1:1, j)))

        fftplan * temp

        map!(imag, view(F, :, j), view(temp, 2:n+1))
    end

    ldiv!(2, F) # up to a scaling
end