module KappaOptimizer

using Random
using Optim
using Statistics

import ..VMCBoseHubbard: estimate_n_max
import ..VMCBoseHubbard: tune_mu_and_log
import ..VMCBoseHubbard: VMC_grand_canonical_adaptive_mu
import ..VMCBoseHubbard: VMC_fixed_particles


export optimize_kappa

function optimize_kappa(sys; N_total::Int, delta_N_tol::Real = 1.0, canonical::Bool=true)

    kappas = Float64[]
    energies = Float64[]
    rejected_kappas = Float64[]
    rejected_nmax = Int[]
    best_result = Ref{Union{Nothing, Tuple}}(nothing)

    function energy_for_kappa_logged(κvec::Vector{Float64}, canonical::Bool=true)
        κ = κvec[1]

        if κ < 0.01 || κ > 10.0
            return 1e6
        end

        if κ <= 0
            @warn "Rejected κ = $κ (non-positive)"
            return Inf
        end

        try
            n_est = estimate_n_max(κ)
            if n_est < 1
                push!(rejected_kappas, κ)
                push!(rejected_nmax, n_est)
                @warn "Rejected κ = $κ due to estimate_n_max = $n_est"
                return Inf
            end

            n_max = clamp(n_est, 1, 30)

            if !canonical
                # --- μ tuning ---
                μ_star, N_final, μ_stats = tune_mu_and_log(sys, κ, n_max, N_total;
                                                        η = 0.02, μ_init = 1.0,
                                                        tune_steps = 5000, walkers = 200)

                if abs(N_final - N_total) > delta_N_tol
                    @warn "Rejected κ = $κ: ⟨N⟩ = $N_final (target = $N_total)"
                    push!(rejected_kappas, κ)
                    push!(rejected_nmax, n_max)
                    return Inf
                end

                println("✅ κ = $κ → μ = $μ_star, ⟨N⟩ = $N_final")

                result = VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_total, μ_star;
                                                        η = 0.0,
                                                        num_walkers = 200,
                                                        num_MC_steps = 2000,
                                                        num_equil_steps = 400)

                vmc_result = result isa Tuple ? result[1] : result
                stats = result isa Tuple ? result[2] : nothing

                E = vmc_result.mean_energy

                push!(kappas, κ)
                push!(energies, E)
                if best_result[] === nothing || E < best_result[][2]
                    best_result[] = (κ, E, vmc_result.sem_energy, μ_star, stats)
                end

            else
                vmc_result = VMC_fixed_particles(sys, κ, n_max, N_total; num_walkers = 200,
                                                        num_MC_steps = 2000,
                                                        num_equil_steps = 400)
                E = vmc_result.mean_energy

                push!(kappas, κ)
                push!(energies, E)
            end

            if !isfinite(E)
                @warn "Non-finite energy at κ = $κ"
                return Inf
            end

            return E

        catch e
            @warn "Error at κ = $κ: $e"
            return Inf
        end
    end

    Random.seed!(42)

    println("🌍 Starting global optimization (Simulated Annealing)...")
    global_result = optimize(energy_for_kappa_logged,
                             [1.0],
                             SimulatedAnnealing(),
                             Optim.Options(iterations = 30, show_trace = false))

    global_kappa = global_result.minimizer[1]
    println("🌍 SA result: κ = $global_kappa, E = $(global_result.minimum)")

    if !isfinite(global_result.minimum)
        @warn "Global optimizer returned Inf energy. Skipping local refinement."
        return (kappa = global_kappa, energy = Inf, sem = Inf, mu = NaN, stats = nothing)
    end

    println("🔍 Starting local refinement...")
    margin = 0.2
    lower_bound = clamp(global_kappa - margin, 0.1, 6.5)
    upper_bound = clamp(global_kappa + margin, 0.1, 6.5)
    initial_kappa = clamp(global_kappa, lower_bound, upper_bound)

    local_result = optimize(energy_for_kappa_logged,
                            [lower_bound], [upper_bound],
                            [initial_kappa],
                            Fminbox(NelderMead()),
                            Optim.Options(iterations = 30, outer_iterations = 30,
                            f_calls_limit = 100, show_trace = false))

    if !isfinite(local_result.minimum)
        @warn "Local refinement failed; falling back to SA result"
    end

    if !canonical

        final_kappa = local_result.minimum < global_result.minimum ? local_result.minimizer[1] : global_kappa
        final_energy = local_result.minimum < global_result.minimum ? local_result.minimizer[2] : global_result.minimum
        println("🔍 Final κ = $final_kappa, Final Energy = $final_energy")

        # Use best_result set inside energy_for_kappa_logged
        if best_result[] === nothing
            return (kappa = NaN, energy = Inf, sem = NaN, mu = NaN, stats = nothing)
        end

        κ_best, E_best, sem_best, μ_best, stats_best = best_result[]
        vmc_result = VMC_grand_canonical_adaptive_mu(sys, κ_best, estimate_n_max(κ_best), N_total, μ_best;
                                                    η = 0.0,
                                                    num_walkers = 1000,
                                                    num_MC_steps = 10000,
                                                    num_equil_steps = 2000)[1]
        return (
            kappa = κ_best,
            energy = E_best,
            sem = sem_best,
            mu = μ_best,
            stats = stats_best,
            E_kin = vmc_result.mean_kinetic,
            E_pot = vmc_result.mean_potential
        )
    else
        final_kappa = Optim.minimizer(local_result)[1]
        final_energy = Optim.minimum(local_result)
        println("🔍 Final κ = $final_kappa, Final Energy = $final_energy")
        vmc_result = VMC_fixed_particles(sys, final_kappa, estimate_n_max(final_kappa), N_total;
                                                    num_walkers = 1000,
                                                    num_MC_steps = 10000,
                                                    num_equil_steps = 2000)
        return (
            kappa = final_kappa,
            energy = vmc_result.mean_energy,
            sem = vmc_result.sem_energy,
            E_kin = vmc_result.mean_kinetic,
            E_kin_sem = vmc_result.sem_kinetic,
            E_pot = vmc_result.mean_potential,
            E_pot_sem = vmc_result.sem_potential
        )
    end
end

end  # module
