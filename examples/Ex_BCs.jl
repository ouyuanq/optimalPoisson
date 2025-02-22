# compare the solving speed among our algorithm, BS algorithm and Kronecker product method
using DelimitedFiles, BenchmarkTools, Printf
include("../src/optimalPoisson.jl")

# u_xx + u_yy = 0 with Neumann and Robin conditions so that the solution is u = 10 * exp(2*x) * cos(2*y)
u = (x, y) -> 10 * exp(2*x) * cos(2*y)
dudx = (x, y) -> 20 * exp(2*x) * cos(2*y)
dudy = (x, y) -> -20 * exp(2*x) * sin(2*y)
theta = 1  # upper Robin boundary condition u(x, 1) + theta*u'(x, 1) = u(x, 1) + theta*dudy(x, 1)

nvec = 2 .^ (5:13)
T = Float64
tol = 1e-14
timevec = zeros(length(nvec), 4)
diffvec_fro = zeros(length(nvec), 4)
# diffvec_inf = zeros(length(nvec), 4)
# diffvec_2 = zeros(length(nvec), 4)
exact = coeffs(y -> cos(2*y), 20) * coeffs(x -> 10 * exp(2*x), 20)'

@printf "Adaptivity to BCs \n"
# our algorithm
for i in eachindex(nvec)
    @printf "No.%d iteration \n" i
    n = nvec[i]

    # set up
    A1, B2, A2, B1, F_ours, T1, T2, g = poisson_mixedbc_init(T, n, n, (x, y) -> zero(T), y -> u(-1, y), y -> dudx(1, y), x -> u(x, -1), x -> u(x, 1) + theta*dudy(x, 1), theta)
    X0_ours = zeros(T, n-2, n-2)
    p_ours, q_ours = ADIshifts(1 / upperboundsecdiffDR(n, theta), 1 / lowerboundsecdiffDR(n, theta), -1 / lowerboundsecdiffDN(n), -1 / upperboundsecdiffDN(n), tol)

    # solution
    begin
        # high resolution
        X_ours = gadi(X0_ours, A1, B2, A2, B1, F_ours, p_ours, q_ours, tol, 10, T1, T2)
        X_ours = bandedlrmul!(Matrix{T}(undef, n, n), T1, X_ours, T2)
        axpy!(true, g, X_ours)
    
        axpy!(-1, exact, view(X_ours, 1:size(exact, 1), 1:size(exact, 2)))
        diffvec_fro[i, 1] = norm(X_ours) / norm(exact)
        # diffvec_inf[i, 1] = norm(X_ours, Inf) / norm(exact, Inf)
        # diffvec_2[i, 1] = opnorm(X_ours) / opnorm(exact)

        # low resolution
        X_ours = gadi(X0_ours, A1, B2, A2, B1, F_ours, p_ours, q_ours, 1e-4, 10, T1, T2)
        X_ours = bandedlrmul!(Matrix{T}(undef, n, n), T1, X_ours, T2)
        axpy!(true, g, X_ours)
    
        axpy!(-1, exact, view(X_ours, 1:size(exact, 1), 1:size(exact, 2)))
        diffvec_fro[i, 2] = norm(X_ours) / norm(exact)
        # diffvec_inf[i, 2] = norm(X_ours, Inf) / norm(exact, Inf)
        # diffvec_2[i, 2] = opnorm(X_ours) / opnorm(exact)
    end

    # timing
    begin
        # high resolution
        ben = @benchmark gadi($(X0_ours), $(A1), $(B2), $(A2), $(B1), $(F_ours), $(p_ours), $(q_ours), $(tol), 10, $(T1), $(T2))
        timevec[i, 1] = minimum(ben).time / 1e9

        # low resolution
        ben = @benchmark gadi($(X0_ours), $(A1), $(B2), $(A2), $(B1), $(F_ours), $(p_ours), $(q_ours), 1e-4, 10, $(T1), $(T2))
        timevec[i, 2] = minimum(ben).time / 1e9
    end
end

# B-S algorithm
for i = 1:8
    @printf "No.%d iteration \n" i
    n = nvec[i]

    # set up
    A1, B2, A2, B1, F_ours, T1, T2, g = poisson_mixedbc_init(T, n, n, (x, y) -> zero(T), y -> u(-1, y), y -> dudx(1, y), x -> u(x, -1), x -> u(x, 1) + theta*dudy(x, 1), theta)

    # solution
    begin
        # X_BS = gsylv(A1, transpose(B2), A2, transpose(B1), F_ours)
        X_BS = sylvester(A2 \ A1, transpose(B2 \ B1), -A2 \ F_ours / transpose(B2))
        X_BS = bandedlrmul!(Matrix{T}(undef, n, n), T1, X_BS, T2)
        axpy!(true, g, X_BS)
    
        axpy!(-1, exact, view(X_BS, 1:size(exact, 1), 1:size(exact, 2)))
        diffvec_fro[i, 3] = norm(X_BS) / norm(exact)
        # diffvec_inf[i, 3] = norm(X_BS, Inf) / norm(exact, Inf)
        # diffvec_2[i, 3] = opnorm(X_BS) / opnorm(exact)
    end

    # timing
    begin
        # ben = @benchmark gsylv($(A1), transpose($(B2)), $(A2), transpose($(B1)), $(F_ours))
        ben = @benchmark sylvester($(A2) \ $(A1), transpose($(B2) \ $(B1)), $(-A2) \ $(F_ours) / transpose($(B2)))
        timevec[i, 3] = minimum(ben).time / 1e9
    end
end

# Townsend & Olver algorithm (need to annotate the warning from LAPACK routine trsyl!)
for i = 1:8
    @printf "No.%d iteration \n" i
    n = nvec[i]

    # Townsend & Olver algorithm
    A1, B2, A2, B1, F, bcy, H, bcx, G = poissonTO_mixedbc_init(T, n, n, (x, y) -> zero(T), y -> u(-1, y), y -> dudx(1, y), x -> u(x, -1), x -> u(x, 1) + theta*dudy(x, 1), theta)
    # solution
    begin
        # X_TO = gsylv(A1, transpose(B2), A2, transpose(B1), F)
        X_TO = sylvester(A2 \ A1, transpose(B2 \ B1), -A2 \ F / transpose(B2))
        X_TO = recoverTO(X_TO, bcy, H, bcx, G)
    
        axpy!(-1, exact, view(X_TO, 1:size(exact, 1), 1:size(exact, 2)))
        diffvec_fro[i, 4] = norm(X_TO) / norm(exact)
        # diffvec_inf[i, 4] = norm(X_TO, Inf) / norm(exact, Inf)
        # diffvec_2[i, 4] = opnorm(X_TO) / opnorm(exact)
    end

    # timing
    begin
        # ben = @benchmark gsylv($(A1), transpose($(B2)), $(A2), transpose($(B1)), $(F))
        ben = @benchmark sylvester($(A2) \ $(A1), transpose($(B2) \ $(B1)), $(-A2) \ $(F) / transpose($(B2)))
        timevec[i, 4] = minimum(ben).time / 1e9
    end
end

open("data/BCs_accuracy.txt", "w") do io
    writedlm(io, [nvec diffvec_fro])
end

open("data/BCs_time.txt", "w") do io
    writedlm(io, [nvec timevec])
end