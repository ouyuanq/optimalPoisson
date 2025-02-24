# an example to show the efficiency of fADI when rhs F has low-rank factorization
include("../src/optimalPoisson.jl")

N = 4  # differential order
T = Float64

exact = (x, y) -> sin(x^2)*exp(y^2)
dexactdx = (x, y) -> 2*x*cos(x^2)*exp(y^2)
dexactdy = (x, y) -> 2*y*sin(x^2)*exp(y^2)
f = (x, y) -> 16*exp(y^2)*(x^4*sin(x^2) - 3*x^2*cos(x^2) + 3*y^2*sin(x^2) + y^4*sin(x^2))

tol = 1e-15

n = 1000
nshift = n

A1, B2, A2, B1, F, Tm, Tn, g = fourth_init(T, n, n, f, (y -> exact(-1, y), y -> dexactdx(-1, y)), (y -> exact(1, y), y -> dexactdx(1, y)), (x -> exact(x, -1), x -> dexactdy(x, -1)), (x -> exact(x, 1), x -> dexactdy(x, 1)))

q, p = ADIshifts(-1 / lowerboundfourthdiffDNDN(nshift), -1 / upperboundfourthdiffDNDN(nshift), 1 / upperboundfourthdiffDNDN(nshift), 1 / lowerboundfourthdiffDNDN(nshift), tol)

svdF = svd(F)
rF = findlast(x -> x > svdF.S[1] * tol, svdF.S)
UF = svdF.U[:, 1:rF] * Diagonal(sqrt.(svdF.S[1:rF]))
VF = svdF.V[:, 1:rF] * Diagonal(sqrt.(svdF.S[1:rF]))

timevec = zeros(length(p), 3)
accuracy = zeros(length(p), 3)
exact_coeffs = coeffs2(exact, n-1, n-1, T)
X0 = zeros(T, n-N, n-N)
@printf "fADI \n"
for i in eachindex(p)
    @printf "Iteration %i \n" i
    pnow = p[1:i]
    qnow = q[1:i]

    # fADI method (type 1)
    Xfadi = gfadi(A1, B2, A2, B1, UF, VF, pnow, qnow, tol, length(pnow)+1)
    Xfadi = Tm * Xfadi * transpose(Tn)
    axpy!(true, g, Xfadi)
    accuracy[i, 1] = norm(Xfadi - exact_coeffs) / norm(exact_coeffs)

    ben = @benchmark gfadi($(A1), $(B2), $(A2), $(B1), $(UF), $(VF), $(pnow), $(qnow), $(tol), length($(pnow))+1)
    timevec[i, 1] = minimum(ben).time / 1e9

    # # fADI method (type 2)
    # Xfadi2 = gfadi2(A1, B2, A2, B1, UF, VF, pnow, qnow, tol, length(pnow)+1)
    # Xfadi2 = Tm * Xfadi2 * transpose(Tn)
    # axpy!(true, g, Xfadi2)
    # accuracy[i, 2] = norm(Xfadi2 - exact_coeffs) / norm(exact_coeffs)

    # ben = @benchmark gfadi2($(A1), $(B2), $(A2), $(B1), $(UF), $(VF), $(pnow), $(qnow), $(tol), length($(pnow))+1)
    # timevec[i, 2] = minimum(ben).time / 1e9
end

@printf "ADI \n"
for i in eachindex(p)
    @printf "Iteration %i \n" i
    pnow = p[1:i]
    qnow = q[1:i]

    # ADI method
    Xadi = gadi(X0, A1, B2, A2, B1, F, pnow, qnow, tol, length(pnow)+1, Tm, Tn)
    Xadi = Tm * Xadi * transpose(Tn)
    axpy!(true, g, Xadi)
    accuracy[i, 3] = norm(Xadi - exact_coeffs) / norm(exact_coeffs)

    ben = @benchmark gadi($(X0), $(A1), $(B2), $(A2), $(B1), $(F), $(pnow), $(qnow), $(tol), length($(pnow))+1, $(Tm), $(Tn))
    timevec[i, 3] = minimum(ben).time / 1e9
end

open("data/fADI.txt", "w") do io
    writedlm(io, [1:length(p) timevec accuracy])
end