using Pkg
Pkg.activate(".")  # Activate project in current folder

include("src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: MC_integration
import ..VMCBoseHubbard: optimize_kappa

# Example run
L = 2
N_target = 2
n_max = 6
U_vals = 1.0:1.0:10.0
t = 1.0
μ_vals = [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.2]
lattice = Lattice1D(L)
grand_canonical = true
projective = true

for U in U_vals
    println("Running U = $U")
    sys = System(t, U, μ, lattice)
    k_opt, history = optimize_kappa(sys, N_target, n_max, grand_canonical, projective;
        κ_init = 1.0,
        η = 0.005,
        num_walkers = 400,
        num_MC_steps = 8_000,
        num_equil_steps = 2_000)
    result = VMC(sys, k_opt, n_max)
    println("κ = $(k_opt), E = $(result.mean_energy)")
end
