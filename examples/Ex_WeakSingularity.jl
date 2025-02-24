# The solution has weak singularities at four corners so that large degree of freedoms are needed for full resolution
# We show that warm restart and reordered shifts are effective
include("../src/optimalPoisson.jl")

nvec = 2 .^ (4:11)
T = Float64
tol = 1e-14
Xiter = zeros(T, 0, 0)
X = zeros(T, 0, 0)
errorvec_cold_identical = Vector{Vector{T}}(undef, 0)
errorvec_cold_reverse = Vector{Vector{T}}(undef, 0)
errorvec_warm_identical = Vector{Vector{T}}(undef, 0)
errorvec_warm_reverse = Vector{Vector{T}}(undef, 0)

@printf "Weak singularity \n"
for i in eachindex(nvec)
    n = nvec[i]

    # set up
    A1, B2, A2, B1, F, T1, T2, g = poisson_init(T, n, n, (x, y) -> zero(T), y -> zero(T), y -> zero(T), x -> zero(T), x -> zero(T))
    F[1, 1] = 1
    # u_xx - 100 * x^2
    A2 = A2 - convertmat(T, n-2, n, 0, 2) * multmatT(n, T.([50; 0; 50])) * T1
    # u_yy + cos(pi*y)
    B2 = B2 + convertmat(T, n-2, n, 0, 2) * multmatT(n, coeffs(y -> cos(pi*y))) * T2

    X0 = zeros(T, n-2, n-2)
    p, q = ADIshifts(1 / (upperboundsecdiffDD(n) + 1), 1 / (lowerboundsecdiffDD(n) - 1), -1 / (lowerboundsecdiffDD(n) - 100), -1 / (upperboundsecdiffDD(n)), tol)

    # solution
    # cold start and identical shifts
    _, errornow, _ = gadi_iter(X0, A1, B2, A2, B1, F, p, q, tol, T1, T2)
    push!(errorvec_cold_identical, errornow)

    # cold start and reversed shifts
    _, errornow, _ = gadi_iter(X0, A1, B2, A2, B1, F, reverse(p), reverse(q), tol, T1, T2)
    push!(errorvec_cold_reverse, errornow)
    
    # warm restart
    copyto!(view(X0, 1:size(Xiter, 1), 1:size(Xiter, 2)), Xiter)

    # warm restart but identical shifts
    _, errornow, _ = gadi_iter(X0, A1, B2, A2, B1, F, p, q, tol, T1, T2)
    push!(errorvec_warm_identical, errornow)

    # reverse the shifts
    if i > 1
        p, q = reverse(p), reverse(q)
    end

    # warm restart and reverse shifts
    global Xiter, errornow, _ = gadi_iter(X0, A1, B2, A2, B1, F, p, q, tol, T1, T2)
    push!(errorvec_warm_reverse, errornow)

    global X = bandedlrmul!(Matrix{T}(undef, n, n), T1, Xiter, T2)
    axpy!(true, g, X)
    if detresolution(X, tol)
        break
    end
end

open("data/weaksingularity.txt", "w") do io
    writedlm(io, [errorvec_cold_identical[4] errorvec_cold_reverse[4] errorvec_warm_identical[4] errorvec_warm_reverse[4]])
end

open("data/weaksingularity_sol.txt", "w") do io
    writedlm(io, X)
end

errorvec_wr = zeros(maximum(length, errorvec_warm_reverse[4:end]), length(errorvec_warm_reverse)-3)
for i = 4:length(errorvec_warm_reverse)
    errorvec_wr[1:length(errorvec_warm_reverse[i]), i-3] = errorvec_warm_reverse[i]
end

open("data/weaksingularity_wr.txt", "w") do io
    writedlm(io, errorvec_wr)
end