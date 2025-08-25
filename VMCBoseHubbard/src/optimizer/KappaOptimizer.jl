module KappaOptimizer

using Random
using Optim
using Statistics

import ..VMCBoseHubbard: estimate_n_max
import ..VMCBoseHubbard: tune_mu_and_log
import ..VMCBoseHubbard: VMC_grand_canonical
import ..VMCBoseHubbard: VMC_fixed_particles


export optimize_kappa

function optimize_kappa(sys, N_total::Int, canonical::Bool; delta_N_tol::Real = 1.0)

    kappas = Float64[]
    energies = Float64[]
    rejected_kappas = Float64[]
    rejected_nmax = Int[]
    best_result = Ref{Union{Nothing, Tuple}}(nothing)

    function energy_for_kappa_logged_canonical(κvec::Vector{Float64})
        κ = κvec[1]

        if κ <= 0
            @warn "Rejected κ = $κ (non-positive)"
            return Inf
        elseif κ >= 3.0
            @warn "Rejected κ = $κ (too large)"
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

            vmc_result = VMC_fixed_particles(sys, κ, n_max, N_total; num_walkers = 200,
                                                    num_MC_steps = 2000,
                                                    num_equil_steps = 400)
            E = vmc_result.mean_energy

            push!(kappas, κ)
            push!(energies, E)

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

    function energy_for_kappa_logged_grand_canonical(κvec::Vector{Float64})
        κ = κvec[1]

        if κ <= 0
            @warn "Rejected κ = $κ (non-positive)"
            return Inf
        elseif κ >= 3.0
            @warn "Rejected κ = $κ (too large)"
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
            n_max = 6

            vmc_result = VMC_grand_canonical(sys, κ, n_max;
                                        N_target = N_total,
                                        num_walkers = 200,
                                        num_MC_steps = 2000,
                                        num_equil_steps = 400)

            E = vmc_result.mean_energy

            if isnan(E)
                @warn "NaN energy detected!"
            elseif isinf(E)
                @warn "Inf energy detected!"
            end

            push!(kappas, κ)
            push!(energies, E)

            # @info "κ = $κ → E = $E"

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


    Random.seed!(2001)

    if !canonical
        println("🌍 Starting global optimization (Simulated Annealing)...")
        global_result = optimize(energy_for_kappa_logged_grand_canonical,
                                [1.0],
                                SimulatedAnnealing(),
                                Optim.Options(iterations = 50, show_trace = false))

        global_kappa = global_result.minimizer[1]
        println("🌍 SA result: κ = $global_kappa, E = $(global_result.minimum)")

        if !isfinite(global_result.minimum)
            @warn "Global optimizer returned Inf energy. Skipping local refinement."
            return (kappa = global_kappa, energy = Inf, sem = Inf, E_kin = Inf, E_kin_sem = Inf, E_pot = Inf, E_pot_sem = Inf, mu = NaN, stats = nothing)
        end

        println("🔍 Starting local refinement...")
        margin = 0.2
        lower_bound = clamp(global_kappa - margin, 0.1, 6.5)
        upper_bound = clamp(global_kappa + margin, 0.1, 6.5)
        initial_kappa = clamp(global_kappa, lower_bound, upper_bound)

        local_result = optimize(energy_for_kappa_logged_grand_canonical,
                                [lower_bound], [upper_bound],
                                [initial_kappa],
                                Fminbox(NelderMead()),
                                Optim.Options(iterations = 5, outer_iterations = 5,
                                f_calls_limit = 5, show_trace = false))

        if !isfinite(local_result.minimum)
            @warn "Local refinement failed; falling back to SA result"
        end

        final_kappa = local_result.minimum < global_result.minimum ? local_result.minimizer[1] : global_kappa
        final_energy = local_result.minimum < global_result.minimum ? local_result.minimum : global_result.minimum
        println("🔍 Final κ = $final_kappa, Final Energy = $final_energy")

        vmc_result = VMC_grand_canonical(sys, final_kappa, #estimate_n_max(final_kappa);
                                                    6;
                                                    N_target = N_total,
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
    else
        println("🌍 Starting global optimization (Simulated Annealing)...")
        global_result = optimize(energy_for_kappa_logged_canonical,
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

        local_result = optimize(energy_for_kappa_logged_canonical,
                                [lower_bound], [upper_bound],
                                [initial_kappa],
                                Fminbox(NelderMead()),
                                Optim.Options(iterations = 10, outer_iterations = 10,
                                f_calls_limit = 10, show_trace = false))

        if !isfinite(local_result.minimum)
            @warn "Local refinement failed; falling back to SA result"
        end

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
