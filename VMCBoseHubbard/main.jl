using Pkg
Pkg.activate(".")  # Activate project in current folder

include("src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

# Example run
L = 12
N = 12
U_vals = 1.0:1.0:3.0
lattice = Lattice1D(L)

for U in U_vals
    println("Running U = $U")
    sys = System(1.0, U, lattice)
    result = optimize_kappa(sys; N_total = N)
    println("κ = $(result.kappa), E = $(result.energy), μ = $(result.mu)")
end
