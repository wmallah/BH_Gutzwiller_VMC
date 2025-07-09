using Pkg
Pkg.activate("../")              # Activate the project environment in the parent directory

include("../src/KappaOptimizer.jl")
using .KappaOptimizer

const BH = KappaOptimizer.VMCBoseHubbard

Lx, Ly = 2, 2
N = 4
t = 1.0
U = [i for i in 1.0:10.0]

# Optimization parameters for smaller VMC runs
num_walkers = 100
num_MC_steps = 1000
num_equil_steps = 200

lattice2D = BH.Lattice2D(Lx, Ly)

results = []

for i in eachindex(U)
    sys = BH.System(t, U[i], lattice2D)
    result = optimize_kappa(sys;
        N_total = N,
        num_walkers = num_walkers,
        num_MC_steps = num_MC_steps,
        num_equil_steps = num_equil_steps)

    push!(results, result)
    # println("κ = $result.kappa: E = $result.energy ± $result.sem")
end

open("../data/2D_VMC_results.dat", "w") do io
    println(io, "# U   kappa   energy   sem")
    for (u, r) in zip(U, results)
        println(io, "$u $(r.kappa) $(r.energy) $(r.sem)")
    end
end
