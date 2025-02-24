# comparisons of solving speed between our algorithm and FT algorithm
using DelimitedFiles, BenchmarkTools, Printf
include("../src/optimalPoisson.jl")

# u_xx + u_yy = -100*x*sin(20*pi*x^2*y)*cos(4*pi*(x+y)) with homogeneous Dirichlet boundary conditions
f = (x, y) -> -100*x*sin(20*pi*x^2*y)*cos(4*pi*(x+y))
nvec = 2 .^ (5:13)
T = Float64
tolvec = [1e-3; 1e-6; 1e-13]
t_ours = zeros(length(nvec), length(tolvec))
t_FT = zeros(length(nvec), 1)
t_FTtrans = zeros(length(nvec), 1)
t_ourstrans = zeros(length(nvec), 1)
diffvec = zeros(length(nvec), length(tolvec))

@printf "Optimality \n"
for i in eachindex(nvec)
    @printf "No.%d iteration \n" i
    n = nvec[i]
    # set up for our algorithm
    A1, B2, A2, B1, F_ours, T1, T2, g = poisson_init(T, n, n, f, y -> zero(T), y -> zero(T), x -> zero(T), x -> zero(T))
    X0_ours = zeros(T, n-2, n-2)

    # set up for Fortunato & Townsend algorithm
    Tm, Tn, F_FT, g = poissonFT_init(T, n, n, f, y -> zero(T), y -> zero(T), x -> zero(T), x -> zero(T))
    X0_FT = Zeros(T, n, n)
    p_FT, q_FT = ADIshifts(-4/pi^2, -39/n^4, 39/n^4, 4/pi^2, 1e-13)
    # @printf "FT iterations: %i \n" length(p_FT)
    X = adi(X0_FT, Tm, Tn, F_FT, p_FT, q_FT)
    X_FT = ultra1mx2cheb(transpose(ultra1mx2cheb(transpose(X))))

    ben = @benchmark adi($(X0_FT), $(Tm), $(Tn), $(F_FT), $(p_FT), $(q_FT))
    t_FT[i] = minimum(ben).time / 1e9

    # ben = @benchmark ultra1mx2cheb(transpose(ultra1mx2cheb(transpose($(X)))))
    # t_FTtrans[i] = minimum(ben).time / 1e9

    for j in eachindex(tolvec)
        @printf "No.%d tolerance \n" j
        tol = tolvec[j]

        p_ours, q_ours = ADIshifts(1 / upperboundsecdiffDD(n), 1 / lowerboundsecdiffDD(n), -1 / lowerboundsecdiffDD(n), -1 / upperboundsecdiffDD(n), tol)
        X = gadi(X0_ours, A1, B2, A2, B1, F_ours, p_ours, q_ours, tol, 10, T1, T2)
        X_ours = bandedlrmul!(Matrix{T}(undef, n, n), T1, X, T2)
        diffvec[i, j] = norm(X_FT - X_ours, Inf)

        ben = @benchmark gadi($(X0_ours), $(A1), $(B2), $(A2), $(B1), $(F_ours), $(p_ours), $(q_ours), $(tol), 10, $(T1), $(T2))
        t_ours[i, j] = minimum(ben).time / 1e9

        # ben = @benchmark bandedlrmul!(Matrix{$(T)}(undef, $(n), $(n)), $(T1), $(X), $(T2))
        # t_ourstrans[i] = minimum(ben).time / 1e9
    end
end

open("data/optimal_time.txt", "w") do io
    writedlm(io, [nvec t_ours t_FT])
end

open("data/optimal_accuracy.txt", "w") do io
    writedlm(io, [nvec diffvec])
end