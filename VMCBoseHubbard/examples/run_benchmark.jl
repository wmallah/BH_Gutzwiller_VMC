using Pkg
Pkg.activate("../")

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard

import ..VMCBoseHubbard: VMC_grand_canonical

# Parameters
L = 12
N_target = 12
t = 1.0
# U_vals = 1.0:1.0:10.0
# U_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
# mu_vals = [1.367650127034, 0.661328990301, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# mu_vals = 1.0:0.1:2.0
U_vals = [9.0]
mu_vals = [1.4]
dim = "1D"
canonical = false  # Set to false for gradient descent VMC

lattice = Lattice1D(L)
ensemble = canonical ? "C" : "GC"
dir_base = "../data/$(ensemble)/$(dim)/L$(L)_N$(N_target)"
mkpath(dir_base)

results = []

for (U, μ) in zip(U_vals, mu_vals)
    println("🔁 Optimizing for U = $U")
    sys = System(t, U, lattice)

    # You may use estimate_n_max(κ_guess) to set this dynamically
    n_max = 8  # Choose conservative upper bound (or use estimate_n_max)

    κ_opt, history = optimize_kappa(sys, n_max, μ;
                                    κ_init = 1.0,
                                    η = 0.05,
                                    N_target = N_target,
                                    num_iters = 40,
                                    num_walkers = 200,
                                    num_MC_steps = 2000)

    # Final evaluation with optimized κ
    final_result = VMC_grand_canonical(sys, κ_opt, n_max;
                                    μ_init = μ,
                                    N_target = N_target,
                                    num_walkers = 400,
                                    num_MC_steps = 10000,
                                    num_equil_steps = 2000)

    push!(results, (U=U, κ=κ_opt, result=final_result))

    hist_file = "$(dir_base)/PN_hist_U$(U).dat"
    open(hist_file, "w") do io
        println(io, "# N  count")
        for (i, count) in enumerate(final_result.PN)
            if count > 0
                N_val = i - 1  # index i corresponds to N = i - 1
                println(io, "$N_val $count")
            end
        end
    end
end

# Save main results
open("$(dir_base)/VMC_results.dat", "w") do io
    println(io, "# U   kappa   energy   sem")
    for entry in results
        r = entry.result
        println(io, "$(entry.U) $(entry.κ) $(r.mean_energy) $(r.sem_energy)")
    end
end

# Save energy parts
open("$(dir_base)/VMC_energy_parts.dat", "w") do io
    println(io, "# U   E_kin    E_kin_sem   E_pot   E_pot_sem")
    for entry in results
        r = entry.result
        println(io, "$(entry.U) $(r.mean_kinetic) $(r.sem_kinetic) $(r.mean_potential) $(r.sem_potential)")
    end
end