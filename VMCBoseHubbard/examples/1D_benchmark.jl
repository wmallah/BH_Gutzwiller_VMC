using Pkg
Pkg.activate("../")  # Activate project environment

include("../src/KappaOptimizer.jl")
using .KappaOptimizer

const BH = KappaOptimizer.VMCBoseHubbard

# Parameters
L = 12
N = 12
t = 1.0
U_vals = 1.0:1.0:10.0

lattice1D = BH.Lattice1D(L)

results = []

for U in U_vals
    println("🔁 Optimizing for U = $U")
    sys = BH.System(t, U, lattice1D)

    result = optimize_kappa(sys; N_total = N)
    push!(results, result)
end

# Save μ and ⟨N⟩ evolution
open("../data/mu_N_vs_step_allU.dat", "w") do io
    println(io, "# U   step   mu   N_avg")
    for (U, r) in zip(U_vals, results)
        if !isnothing(r.stats)
            for (s, mu, N) in zip(r.stats.steps, r.stats.mu_trace, r.stats.N_trace)
                println(io, "$U $s $mu $N")
            end
        end
    end
end

# Save total energy summary
open("../data/1D_VMC_results.dat", "w") do io
    println(io, "# U   kappa   energy   sem   mu")
    for (U, r) in zip(U_vals, results)
        println(io, "$U $(r.kappa) $(r.energy) $(r.sem) $(r.mu)")
    end
end

# Save kinetic and potential energy components
open("../data/1D_VMC_energy_parts.dat", "w") do io
    println(io, "# U   E_kin   E_pot")
    for (U, r) in zip(U_vals, results)
        println(io, "$U $(r.E_kin) $(r.E_pot)")
    end
end
