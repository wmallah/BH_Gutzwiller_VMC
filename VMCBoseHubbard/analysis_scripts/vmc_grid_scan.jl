using Printf, Optim, Random
include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

L = 12
N_total = 12
t = 1.0
U = 3.3578

lattice = Lattice1D(L)
sys = System(t, U, lattice)

output_file = open("vmc_grid_scan.dat", "w")
# Header line
println(output_file, "# U     kappa     energy      sem")

kappa_values = 0.4:0.05:2.0

for κ in kappa_values
    try
        n_max = estimate_n_max(κ)
        result = run_vmc(sys, κ, n_max, N_total;
                         num_walkers=300,
                         num_MC_steps=3000,
                         num_equil_steps=500)
        @printf(output_file, "%5.4f  %7.3f  %10.6f  %10.6f\n", U, κ, result.mean_energy, result.sem_energy)
    catch e
        @warn "Failed at κ=$κ: $e"
    end
end

close(output_file)
