# optimal complexity Poisson solver for spectral method
using LinearAlgebra, FFTW, BandedMatrices, SemiseparableMatrices, MatrixEquations, FillArrays, ArrayLayouts, BenchmarkTools, Printf, DelimitedFiles, Plots
BLAS.set_num_threads(1)

include("Operators.jl")
include("Chebyshev.jl")
include("Interpolation.jl")
include("ADIshifts.jl")
include("Banded.jl")
include("ADI.jl")
include("Poisson.jl")
include("Fourth.jl")
include("Cheb2ultra.jl")
include("Chebop2.jl")
include("FADI.jl")