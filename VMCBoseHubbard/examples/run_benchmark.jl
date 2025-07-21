using Pkg
Pkg.activate("../")  # Activate the project environment at the root

# Load the full VMCBoseHubbard module (this includes KappaOptimizer automatically)
include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

# Parameters
# Lx, Ly = 4, 4
L = 8
N = 16
t = 1.0
U_vals = 1.0:1.0:10.0

# Select lattice and dimension
lattice = Lattice1D(L)
dim = "1D"  # "1D"  # or "2D" if using Lattice2D
canonical = true  # or false for grand canonical

# Create output directory label
ensemble = canonical ? "C" : "GC"
dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N)"
mkpath(dir_base)  # Ensure the directory exists

results = []

for U in U_vals
    println("🔁 Optimizing for U = $U")
    sys = System(t, U, lattice)
    result = optimize_kappa(sys; N_total = canonical ? N : nothing)
    push!(results, result)
end

# Save μ and ⟨N⟩ traces if grand canonical
if !canonical
    open("$(dir_base)/mu_N_vs_step_allU.dat", "w") do io
        println(io, "# U   step   mu   N_avg")
        for (U, r) in zip(U_vals, results)
            if !isnothing(r.stats)
                for (s, mu, N) in zip(r.stats.steps, r.stats.mu_trace, r.stats.N_trace)
                    println(io, "$U $s $mu $N")
                end
            end
        end
    end

    open("$(dir_base)/VMC_results.dat", "w") do io
        println(io, "# U   kappa   energy   sem   mu")
        for (U, r) in zip(U_vals, results)
            println(io, "$U $(r.kappa) $(r.energy) $(r.sem) $(r.mu)")
        end
    end
else
    open("$(dir_base)/VMC_results.dat", "w") do io
        println(io, "# U   kappa   energy   sem")
        for (U, r) in zip(U_vals, results)
            println(io, "$U $(r.kappa) $(r.energy) $(r.sem)")
        end
    end
end

# Save kinetic and potential energy parts
open("$(dir_base)/VMC_energy_parts.dat", "w") do io
    println(io, "# U   E_kin    E_kin_sem   E_pot   E_pot_sem")
    for (U, r) in zip(U_vals, results)
        println(io, "$U $(r.E_kin) $(r.E_kin_sem) $(r.E_pot) $(r.E_pot_sem)")
    end
end
