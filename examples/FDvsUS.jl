# compare the accuracy and speed of finite difference method and the new method when solving a Poisson equation with homogeneous Dirichlet boundary conditions
include("../src/optimalPoisson.jl")
# u_xx + u_yy = 1, u(±1, y) = u(x, ±1) = 0

T = Float64

# the reference solution
nref = 2048
tolref = 1e-15
A1ref, B2ref, A2ref, B1ref, Fref, T1ref, T2ref, gref = poisson_init(T, nref, nref, (x, y) -> zero(T), y -> zero(T), y -> zero(T), x -> zero(T), x -> zero(T))
Fref[1, 1] = 1

X0ref = zeros(T, nref - 2, nref - 2)
pref, qref = ADIshifts(1 / upperboundsecdiffDD(nref), 1 / lowerboundsecdiffDD(nref), -1 / lowerboundsecdiffDD(nref), -1 / upperboundsecdiffDD(nref), tolref)

# no error checking
X_ref, it_ref = gadi(X0ref, A1ref, B2ref, A2ref, B1ref, Fref, pref, qref, tolref, 100, T1ref, T2ref)
X_true = bandedlrmul!(Matrix{T}(undef, nref, nref), T1ref, X_ref, T2ref)


# US method
nvec = 2 .^ (3:10)
accvec_US = zeros(T, length(nvec))
timevec_US = zeros(T, length(nvec))
X_grid = evaluate(X_true, chebpts(nvec[end], T), chebpts(nvec[end], T))
tol = 1e-15
@printf "US method \n"
for i in eachindex(nvec)
    @printf "No.%d iteration \n" i
    n = nvec[i]
    # set up
    A1, B2, A2, B1, F, T1, T2, g = poisson_init(T, n, n, (x, y) -> zero(T), y -> zero(T), y -> zero(T), x -> zero(T), x -> zero(T))
    F[1, 1] = 1

    X0 = zeros(T, n - 2, n - 2)
    p, q = ADIshifts(1 / upperboundsecdiffDD(n), 1 / lowerboundsecdiffDD(n), -1 / lowerboundsecdiffDD(n), -1 / upperboundsecdiffDD(n), tol)

    X_adi, _ = gadi(X0, A1, B2, A2, B1, F, p, q, tol, 10, T1, T2)
    X_i = bandedlrmul!(Matrix{T}(undef, n, n), T1, X_adi, T2)
    accvec_US[i] = norm(X_grid - evaluate(X_i, chebpts(nvec[end], T), chebpts(nvec[end], T)))

    ben = @benchmark gadi($(X0), $(A1), $(B2), $(A2), $(B1), $(F), $(p), $(q), $(tol), 10, $(T1), $(T2))
    timevec_US[i] = minimum(ben).time / 1e9
end

open("data/US.txt", "w") do io
    writedlm(io, [accvec_US timevec_US])
end

# FD method
nvec = 2 .^ (4:14) .- 1
accvec_FD = zeros(T, length(nvec))
timevec_FD = zeros(T, length(nvec))
@printf "FD method \n"
for i in eachindex(nvec)
    @printf "No.%d iteration \n" i
    n = nvec[i]
    F = ones(T, n, n)
    
    X_FD = Poisson_FD(F)
    accvec_FD[i] = norm(evaluate(X_true, range(-1, 1, length=n+2)[2:end-1], range(-1, 1, length=n+2)[2:end-1]) - X_FD)

    ben = @benchmark Poisson_FD($(F))
    timevec_FD[i] = minimum(ben).time / 1e9
end

open("data/FD.txt", "w") do io
    writedlm(io, [accvec_FD timevec_FD])
end