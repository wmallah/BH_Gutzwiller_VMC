module KappaOptimizer

using Optim, Random, Statistics
include("VMCBoseHubbard.jl")
using .VMCBoseHubbard

export optimize_kappa

function optimize_kappa(sys::System; N_total, kwargs...)

    kappas = Float64[]
    energies = Float64[]
    rejected_kappas = Float64[]
    rejected_nmax = Int[]

    function energy_for_kappa_logged(κvec::Vector{Float64}, sys::System; kwargs...)
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

            n_max = clamp(n_est, 1, 30)  # allow n_max down to 1



            result = run_vmc(sys, κ, n_max, N_total; kwargs...)
            E = result.mean_energy

            if E == 0.0 || !isfinite(E)
                @warn "Suspicious or non-finite energy at κ = $κ → E = $E"
                return Inf
            end

            push!(kappas, κ)
            push!(energies, E)
            return E

        catch e
            @warn "Error at κ = $κ: $e"
            return Inf
        end
    end

    function bounded_energy_for_kappa(κvec::Vector{Float64}, sys::System; kwargs...)
        κ = κvec[1]
        if κ < 0.1 || κ > 3.0
            return 1e6 + abs(κ - 3)^2   # provide large penalty instead of inf to deter optimizer from that region but not fully shut down optimizer
        end
        return energy_for_kappa_logged(κvec, sys; kwargs...)
    end

    Random.seed!(42)

    println("🌍 Starting global optimization (Simulated Annealing)...")
    global_result = optimize(x -> bounded_energy_for_kappa(x, sys; kwargs...),
                            [1.0],
                            SimulatedAnnealing(),
                            Optim.Options(iterations = 100, show_trace = false))

    global_kappa = global_result.minimizer[1]
    println("🌍 SA result: κ = $global_kappa, E = $(global_result.minimum)")

    # Ensure the global optimizer didn't return an Inf energy, setting up the local optimizer for failure
    if !isfinite(global_result.minimum)
        @warn "Global optimizer returned Inf energy. Skipping local refinement."
        return (kappa = global_kappa, energy = Inf, sem = Inf)
    end

    println("\n🔍 Starting local refinement from SA result...")
    # Adaptive bounding based on SA result
    margin = 0.2
    lower_bound = clamp(global_kappa - margin, 0.1, 6.5)
    upper_bound = clamp(global_kappa + margin, 0.1, 6.5)
    initial_kappa = clamp(global_kappa, lower_bound, upper_bound)


    if upper_bound - lower_bound < 1e-3
        upper_bound += 0.1
        lower_bound = max(lower_bound - 0.1, 1e-4)
    end

    local_result = optimize(x -> energy_for_kappa_logged(x, sys; kwargs...),
                            [lower_bound], [upper_bound],
                            [initial_kappa],
                            Fminbox(NelderMead()),
                            Optim.Options(iterations = 100, show_trace = false))

    # If local refinement returns an Inf energy, retry with expanded bounds
    if !isfinite(local_result.minimum)
        @warn "Local refinement failed; retrying with expanded margin..."
        new_margin = 0.5
        lower_bound = clamp(global_kappa - new_margin, 0.1, 6.5)
        upper_bound = clamp(global_kappa + new_margin, 0.1, 6.5)
        initial_kappa = clamp(global_kappa, lower_bound, upper_bound)

        local_result = optimize(x -> energy_for_kappa_logged(x, sys; kwargs...),
                                [lower_bound], [upper_bound],
                                [initial_kappa],
                                Fminbox(NelderMead()),
                                Optim.Options(iterations = 100, show_trace = false))
    end

    # If local refinement still returns Inf after retry with expanded bounds, fall back on global result
    if !isfinite(local_result.minimum)
        @warn "Local optimizer failed entirely — falling back to SA result"
        refined_kappa = global_kappa
        refined_energy = global_result.minimum
    else
        refined_kappa = local_result.minimizer[1]
        refined_energy = local_result.minimum
    end

    println("🔍 Refined result: κ = $refined_kappa")
    println("🏁 Refined energy = $refined_energy")

    n_max_final = estimate_n_max(refined_kappa)
    final_result = run_vmc(sys, refined_kappa, n_max_final, N_total; num_walkers=1000, num_MC_steps=10000, num_equil_steps=2000)

    println("📊 Final confirmed energy = $(final_result.mean_energy) ± $(final_result.sem_energy)")

    if !isempty(rejected_kappas)
        println("\n⚠️  Rejected κ values:")
        for (k, n) in zip(rejected_kappas, rejected_nmax)
            println("  κ = $k → estimate_n_max = $n")
        end
    end

    return (kappa = refined_kappa,
            energy = final_result.mean_energy,
            sem = final_result.sem_energy)
end

end  # module