using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: VMC

# -----------------------
# System parameters
# -----------------------
L = 4
N_target = 2
t = 1.0

# 12x12 mu values
U_vals = [1.0]
μ_vals = [-1.996641490650]

# μ_vals = [0.0, 1.5, 4.2]

dim = "1D"
grand_canonical = false
projective = false

lattice = Lattice1D(L)
ensemble = !grand_canonical ? "C" : "GC"

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
    n_max = 18

    # -----------------------
    # Optimize κ (MC-error stopping)
    # -----------------------
    κ_opt, history = optimize_kappa(
        sys, N_target, n_max, μ, grand_canonical, projective;
        κ_init = 1.0,
        η = 0.05,
        num_walkers = 400,
        num_MC_steps = 8000,
        num_equil_steps = 2000
    )

    println("    Optimal κ = $(round(κ_opt, digits=10))")

    # -----------------------
    # Final high-statistics evaluation
    # -----------------------
    final_result = VMC(
        sys, N_target, κ_opt, n_max, μ, grand_canonical, projective;
        num_walkers = 200,
        num_MC_steps = 50_000,
        num_equil_steps = 10_000,
    )

    acceptance_ratio = final_result.acceptance_ratio

    println("Acceptance Ratio: $acceptance_ratio")

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
