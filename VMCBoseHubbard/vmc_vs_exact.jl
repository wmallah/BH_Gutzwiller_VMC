using Pkg
Pkg.activate(".")              # Activate the project environment in the parent directory

using VMCBoseHubbard
using Statistics
using Printf
using DelimitedFiles

L = 2
N = 2
t = 1.0
κ = 0.3
n_max = 4
num_walkers = 200
num_MC_steps = 2000
num_equil_steps = 500

U_vals = 0.5:0.5:10.0
E_exact = Float64[]
E_vmc = Float64[]
E_err = Float64[]

for U in U_vals
    # Exact energy: E = (U - sqrt(U^2 + 16t^2)) / 2
    push!(E_exact, (U - sqrt(U^2 + 16t^2)) / 2)

    lattice = Lattice1D(L)
    sys = System(t, U, lattice)
    res = run_vmc(sys, κ, n_max; num_walkers=num_walkers, num_MC_steps=num_MC_steps, num_equil_steps=num_equil_steps)
    push!(E_vmc, res.mean_energy)
    push!(E_err, res.sem_energy)
end

writedlm("vmc_results.csv", [collect(U_vals) E_exact E_vmc E_err], ',')
