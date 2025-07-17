module KappaOptimizer

using Optim, Random, Statistics
include("VMCBoseHubbard.jl")
using .VMCBoseHubbard

export optimize_kappa

function optimize_kappa(sys::System; N_total::Int, delta_N_tol::Real = 1.0)

    kappas = Float64[]
    energies = Float64[]
    rejected_kappas = Float64[]
    rejected_nmax = Int[]
    best_result = Ref{Union{Nothing, Tuple}}(nothing)

    function energy_for_kappa_logged(κvec::Vector{Float64})
        κ = κvec[1]

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
                                                      num_walkers = 1000,
                                                      num_MC_steps = 10000,
                                                      num_equil_steps = 2000)

            vmc_result = result isa Tuple ? result[1] : result
            stats = result isa Tuple ? result[2] : nothing

            E = vmc_result.mean_energy
            if !isfinite(E)
                @warn "Non-finite energy at κ = $κ"
                return Inf
            end

            push!(kappas, κ)
            push!(energies, E)
            if best_result[] === nothing || E < best_result[][2]
                best_result[] = (κ, E, vmc_result.sem_energy, μ_star, stats)
            end


            return E

        catch e
            @warn "Error at κ = $κ: $e"
            return Inf
        end
    end

    function bounded_energy_for_kappa(κvec::Vector{Float64})
        κ = κvec[1]
        if κ < 0.01 || κ > 5.0
            return 1e6 + abs(κ - 3)^2
        end
        return energy_for_kappa_logged(κvec)
    end

    Random.seed!(42)

    println("🌍 Starting global optimization (Simulated Annealing)...")
    global_result = optimize(bounded_energy_for_kappa,
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

    local_result = optimize(bounded_energy_for_kappa,
                            [lower_bound], [upper_bound],
                            [initial_kappa],
                            Fminbox(NelderMead()),
                            Optim.Options(iterations = 30, show_trace = false))

    if !isfinite(local_result.minimum)
        @warn "Local refinement failed; falling back to SA result"
    end

    final_kappa = local_result.minimum < global_result.minimum ? local_result.minimizer[1] : global_kappa

    println("🔍 Final κ = $final_kappa")

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

end

end  # module
