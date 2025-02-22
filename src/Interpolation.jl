# low-order polynomial interpolation for boundary conditions on the square [-1,1]x[-1,1]

## second-order equations
# Dirichlet conditions
function interpDD(lbc::Vector{T}, rbc::Vector{T}, dbc::Vector{T}, ubc::Vector{T}, m::Integer, n::Integer) where T
    # Chebyshev coefficients of boundary conditions are given in lbc, rbc, dbc and ubc respectively
    g = zeros(T, m, n)

    # interpolation on left and right BCs
    llbc, lrbc = min(m, length(lbc)), min(m, length(rbc))
    broadcast!(+, view(g, 1:lrbc, 1:2), view(g, 1:lrbc, 1:2), view(rbc, 1:lrbc))
    axpy!(true, view(lbc, 1:llbc), view(g, 1:llbc, 1))
    axpy!(-1, view(lbc, 1:llbc), view(g, 1:llbc, 2))

    # corrections for down and upper BCs
    axpy!(-1, sum(view(g, 1:2:max(llbc, lrbc), 1:2), dims=1), view(g, 1, 1:2))
    axpy!(-1, sum(view(g, 2:2:max(llbc, lrbc), 1:2), dims=1), view(g, 2, 1:2))

    # add the down and upper BCs
    ldbc, lubc = min(n, length(dbc)), min(n, length(ubc))
    tg = transpose(g)
    broadcast!(+, view(tg, 1:lubc, 1:2), view(tg, 1:lubc, 1:2), view(ubc, 1:lubc))
    axpy!(true, view(dbc, 1:ldbc), view(tg, 1:ldbc, 1))
    axpy!(-1, view(dbc, 1:ldbc), view(tg, 1:ldbc, 2))

    # division by 2
    ldiv!(2, view(g, 1:max(llbc, lrbc), 1:2))
    ldiv!(2, view(g, 1:2, 3:max(ldbc, lubc)))

    g
end

# left Dirichlet and right Neumann conditions
function interpDN(lbc::Vector{T}, rbc::Vector{T}, dbc::Vector{T}, ubc::Vector{T}, m::Integer, n::Integer) where T
    # Chebyshev coefficients of boundary conditions are given in lbc, rbc, dbc and ubc respectively
    ly, lx = max(length(lbc), length(rbc)), max(length(dbc), length(ubc))
    @assert m >= ly && n >= lx "Boundary conditions are not resolved"
    g = zeros(T, m, n)

    # conversion matrix from Hermite basis to ChebyshevT basis
    N2T = [8 -9 0 1; -2 -1 2 1] ./ 16

    # interpolation on left and right BCs
    mul!(view(g, 1:length(lbc), 1:4), lbc, view(N2T, 1:1, :), true, true)
    mul!(view(g, 1:length(rbc), 1:4), rbc, view(N2T, 2:2, :), true, true)

    # corrections for down and upper BCs
    axpy!(true, transpose(sum(view(g, 2:2:ly, 1:4), dims=1) - sum(view(g, 1:2:ly, 1:4), dims=1)), view(dbc, 1:4))
    dg = broadcast(*, 0:ly-1, view(g, 1:ly, 1:4))  # Neumann coeffs of y-interpolation
    broadcast!(*, dg, 0:ly-1, dg)
    axpy!(-1, transpose(sum(view(dg, 1:ly, 1:4), dims=1)), view(ubc, 1:4))

    # add the down and upper BCs
    tg = transpose(g)
    mul!(view(tg, 1:length(dbc), 1:4), dbc, view(N2T, 1:1, :), true, true)
    mul!(view(tg, 1:length(ubc), 1:4), ubc, view(N2T, 2:2, :), true, true)

    g
end

# left Dirichlet and right Robin conditions
function interpDR(theta::Number, lbc::Vector{T}, rbc::Vector{T}, dbc::Vector{T}, ubc::Vector{T}, m::Integer, n::Integer) where T
    # Chebyshev coefficients of boundary conditions are given in lbc, rbc, dbc and ubc respectively
    ly, lx = max(length(lbc), length(rbc)), max(length(dbc), length(ubc))
    @assert m >= ly && n >= lx "Boundary conditions are not resolved"
    g = zeros(T, m, n)

    # conversion matrix from Hermite basis to ChebyshevT basis
    R2T_left = [8 -9 0 1] ./ 16
    R2T_right = [8 9 0 -1] .* (theta/(16 * (1+theta))) .+ [-2 -1 2 1] .* (1/(16*(1+theta)*theta))

    # interpolation on left and right BCs
    mul!(view(g, 1:length(lbc), 1:4), lbc, R2T_left, true, true)
    mul!(view(g, 1:length(rbc), 1:4), rbc, R2T_right, true, true)

    # corrections for down and upper BCs
    axpy!(true, transpose(sum(view(g, 2:2:ly, 1:4), dims=1) - sum(view(g, 1:2:ly, 1:4), dims=1)), view(dbc, 1:4))
    dg = broadcast(*, 0:ly-1, view(g, 1:ly, 1:4))  # Neumann coeffs of y-interpolation
    broadcast!(*, dg, 0:ly-1, dg)
    axpy!(-1, transpose(sum(view(g, 1:ly, 1:4), dims=1)), view(ubc, 1:4))
    axpy!(-theta, transpose(sum(view(dg, 1:ly, 1:4), dims=1)), view(ubc, 1:4))

    # add the down and upper BCs
    tg = transpose(g)
    mul!(view(tg, 1:length(dbc), 1:4), dbc, R2T_left, true, true)
    mul!(view(tg, 1:length(ubc), 1:4), ubc, R2T_right, true, true)

    g
end

# left Dirichlet, right Neumann, down Dirichlet and upper Robin conditions
function interp_mixed(theta::Number, lbc::Vector{T}, rbc::Vector{T}, dbc::Vector{T}, ubc::Vector{T}, m::Integer, n::Integer) where T
    # Chebyshev coefficients of boundary conditions are given in lbc, rbc, dbc and ubc respectively
    ly, lx = max(length(lbc), length(rbc)), max(length(dbc), length(ubc))
    @assert m >= ly && n >= lx "Boundary conditions are not resolved"
    g = zeros(T, m, n)

    # conversion matrix from Hermite basis to ChebyshevT basis
    N2T = [8 -9 0 1; -2 -1 2 1] ./ 16
    R2T_left = [8 -9 0 1] ./ 16
    R2T_right = [8 9 0 -1] .* (theta/(16 * (1+theta))) .+ [-2 -1 2 1] .* (1/(16*(1+theta)*theta))

    # interpolation on left and right BCs
    mul!(view(g, 1:length(lbc), 1:4), lbc, view(N2T, 1:1, :), true, true)
    mul!(view(g, 1:length(rbc), 1:4), rbc, view(N2T, 2:2, :), true, true)

    # corrections for down and upper BCs
    axpy!(true, transpose(sum(view(g, 2:2:ly, 1:4), dims=1) - sum(view(g, 1:2:ly, 1:4), dims=1)), view(dbc, 1:4))
    dg = broadcast(*, 0:ly-1, view(g, 1:ly, 1:4))  # Neumann coeffs of y-interpolation
    broadcast!(*, dg, 0:ly-1, dg)
    axpy!(-1, transpose(sum(view(g, 1:ly, 1:4), dims=1)), view(ubc, 1:4))
    axpy!(-theta, transpose(sum(view(dg, 1:ly, 1:4), dims=1)), view(ubc, 1:4))

    # add the down and upper BCs
    tg = transpose(g)
    mul!(view(tg, 1:length(dbc), 1:4), dbc, R2T_left, true, true)
    mul!(view(tg, 1:length(ubc), 1:4), ubc, R2T_right, true, true)

    g
end

## fourth-order equations
# Dirichlet and Neumann conditions
function interpDNDN(lbc::NTuple{2, Vector{T}}, rbc::NTuple{2, Vector{T}}, dbc::NTuple{2, Vector{T}}, ubc::NTuple{2, Vector{T}}, m::Integer, n::Integer) where T
    # Chebyshev coefficients of Dirichlet and Neumann boundary conditions are given in lbc, rbc, dbc and ubc respectively
    ly, lx = max(maximum(length, lbc), maximum(length, rbc)), max(maximum(length, dbc), maximum(length, ubc))
    @assert m >= ly && n >= lx "Boundary conditions are not resolved"
    g = zeros(T, m, n)

    # conversion matrix from Hermite basis to ChebyshevT basis
    H2T = [8 -9 0 1; 2 -1 -2 1; 8 9 0 -1; -2 -1 2 1] ./ 16

    # interpolation on left and right BCs
    mul!(view(g, 1:length(lbc[1]), 1:4), lbc[1], view(H2T, 1:1, :), true, true)
    mul!(view(g, 1:length(lbc[2]), 1:4), lbc[2], view(H2T, 2:2, :), true, true)
    mul!(view(g, 1:length(rbc[1]), 1:4), rbc[1], view(H2T, 3:3, :), true, true)
    mul!(view(g, 1:length(rbc[2]), 1:4), rbc[2], view(H2T, 4:4, :), true, true)

    # corrections for down and upper BCs
    axpy!(true, transpose(sum(view(g, 2:2:ly, 1:4), dims=1) - sum(view(g, 1:2:ly, 1:4), dims=1)), view(dbc[1], 1:4))
    axpy!(-1, transpose(sum(view(g, 1:ly, 1:4), dims=1)), view(ubc[1], 1:4))
    dg = broadcast(*, 0:ly-1, view(g, 1:ly, 1:4))  # Neumann coeffs of y-interpolation
    broadcast!(*, dg, 0:ly-1, dg)
    axpy!(true, transpose(sum(view(dg, 1:2:ly, 1:4), dims=1) - sum(view(dg, 2:2:ly, 1:4), dims=1)), view(dbc[2], 1:4))
    axpy!(-1, transpose(sum(view(dg, 1:ly, 1:4), dims=1)), view(ubc[2], 1:4))

    # add the down and upper BCs
    tg = transpose(g)
    mul!(view(tg, 1:length(dbc[1]), 1:4), dbc[1], view(H2T, 1:1, :), true, true)
    mul!(view(tg, 1:length(dbc[2]), 1:4), dbc[2], view(H2T, 2:2, :), true, true)
    mul!(view(tg, 1:length(ubc[1]), 1:4), ubc[1], view(H2T, 3:3, :), true, true)
    mul!(view(tg, 1:length(ubc[2]), 1:4), ubc[2], view(H2T, 4:4, :), true, true)

    g
end