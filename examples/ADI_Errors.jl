include("../src/optimalPoisson.jl")

nvec = 2 .^ (8:11)
T = Float64
tol = eps(T)
u = (x, y) -> 10 * exp(2*x) * cos(2*y)
exact = coeffs(y -> cos(2*y), 20) * coeffs(x -> 10 * exp(2*x), 20)'
trueerrvec = Vector{Vector{T}}(undef, 0)

for i in eachindex(nvec)
    n = nvec[i]
    A1, B2, A2, B1, F, T1, T2, g = poisson_init(T, n, n, (x, y) -> zero(T), y -> u(-1, y), y -> u(1, y), x -> u(x, -1), x -> u(x, 1))
    p, q = ADIshifts(1 / upperboundsecdiffDD(n), 1 / lowerboundsecdiffDD(n), -1 / lowerboundsecdiffDD(n), -1 / upperboundsecdiffDD(n), tol)

    X0 = zeros(T, n-2, n-2)
    _, Cauchyerror, trueerror = gadi_iter(X0, A1, B2, A2, B1, F, p, q, tol, T1, T2; truesol = axpy!(true, exact, -g[1:21, 1:21]))

    push!(trueerrvec, trueerror)
end

results = zeros(T, maximum(length, trueerrvec) + 1, length(nvec))
for i in eachindex(nvec)
    leni = length(trueerrvec[i])
    results[1, i] = leni
    results[2:leni+1, i] = trueerrvec[i]
end

open("data/ADI_Errors.txt", "w") do io
    writedlm(io, results)
end