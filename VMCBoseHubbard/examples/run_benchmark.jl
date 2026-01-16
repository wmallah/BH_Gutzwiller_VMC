using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: VMC_grand_canonical

# -----------------------
# System parameters
# -----------------------
L = 2
N_target = 2
t = 1.0

U_vals = [1.0]
μ_vals = [0.0]

dim = "1D"
canonical = false   # GC ensemble for optimization

lattice = Lattice1D(L)
ensemble = canonical ? "C" : "GC"

dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)"
mkpath(dir_base)

results = []

# -----------------------
# Loop over parameters
# -----------------------
for (U, μ) in zip(U_vals, μ_vals)

    println("Optimizing Gutzwiller κ for U = $U, μ = $μ")

    sys = System(t, U, lattice)

    # Conservative truncation
    n_max = 8

    # -----------------------
    # Optimize κ (MC-error stopping)
    # -----------------------
    κ_opt, history = optimize_kappa(
        sys, n_max, μ;
        κ_init = 1.0,
        η = 0.05,
        N_target = N_target,
        num_walkers = 200,
        num_MC_steps = 4000,
        num_equil_steps = 1000
    )

    println("    Optimal κ = $(round(κ_opt, digits=10))")

    # -----------------------
    # Final high-statistics evaluation
    # -----------------------
    final_result = VMC_grand_canonical(
        sys, κ_opt, n_max, μ, N_target;
        num_walkers = 200,
        num_MC_steps = 50_000,
        num_equil_steps = 10_000
    )

    push!(results, (U = U, κ = κ_opt, result = final_result))

    # -----------------------
    # Save particle-number histogram
    # -----------------------
    hist_file = "$(dir_base)/PN_hist_U$(U).dat"
    open(hist_file, "w") do io
        println(io, "# N   count")
        for (i, count) in enumerate(final_result.PN)
            if count > 0
                println(io, "$(i - 1) $count")
            end
        end
    end
end

# -----------------------
# Save total energies
# -----------------------
open("$(dir_base)/VMC_results.dat", "w") do io
    println(io, "# U   kappa   energy   sem")
    for entry in results
        r = entry.result
        println(io, "$(entry.U) $(entry.κ) $(r.mean_energy) $(r.sem_energy)")
    end
end

# -----------------------
# Save energy components
# -----------------------
open("$(dir_base)/VMC_energy_parts.dat", "w") do io
    println(io, "# U   E_kin   E_kin_sem   E_pot   E_pot_sem")
    for entry in results
        r = entry.result
        println(io,
            "$(entry.U) " *
            "$(r.mean_kinetic) $(r.sem_kinetic) " *
            "$(r.mean_potential) $(r.sem_potential)"
        )
    end
end
