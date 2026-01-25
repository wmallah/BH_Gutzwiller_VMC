using Pkg
Pkg.activate(".")  # Activate project in current folder

include("src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: VMC

# Example run
L = 12
N = 12
n_max = 6
U_vals = 1.0:1.0:3.0
lattice = Lattice1D(L)

for U in U_vals
    println("Running U = $U")
    sys = System(1.0, U, lattice)
    k_opt, history = optimize_kappa(sys, n_max; N_target = N)
    result = VMC(sys, k_opt, n_max)
    println("κ = $(k_opt), E = $(result.mean_energy)")
end
