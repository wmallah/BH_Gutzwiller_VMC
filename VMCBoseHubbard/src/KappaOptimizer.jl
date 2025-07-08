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
            n_max = estimate_n_max(κ)

            if n_max < 3 || n_max > 30
                push!(rejected_kappas, κ)
                push!(rejected_nmax, n_max)
                @warn "Rejected κ = $κ due to estimate_n_max = $n_max"
                return Inf
            end

            result = run_vmc(sys, κ, n_max, N_total; kwargs...)
            E = result.mean_energy

            if !isfinite(E)
                @warn "Non-finite energy at κ = $κ"
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

    Random.seed!(42)

    println("🌍 Starting global optimization (Simulated Annealing)...")
    global_result = optimize(x -> energy_for_kappa_logged(x, sys; kwargs...),
                             [1.0],
                             SimulatedAnnealing(),
                             Optim.Options(iterations = 10, show_trace = true))

    global_kappa = global_result.minimizer[1]
    println("🌍 SA result: κ = $global_kappa, E = $(global_result.minimum)")

    println("\n🔍 Starting local refinement from SA result...")
    lower_bound = 0.5
    upper_bound = 2.0
    initial_kappa = clamp(global_kappa, lower_bound, upper_bound)

    local_result = optimize(x -> energy_for_kappa_logged(x, sys; kwargs...),
                            [lower_bound], [upper_bound],
                            [initial_kappa],
                            Fminbox(NelderMead()),
                            Optim.Options(iterations = 10, show_trace = true))

    refined_kappa = local_result.minimizer[1]
    refined_energy = local_result.minimum
    # println("🔍 Refined result: κ = $refined_kappa")
    # println("🏁 Refined energy = $refined_energy")

    n_max_final = estimate_n_max(refined_kappa)
    final_result = run_vmc(sys, refined_kappa, n_max_final, N_total; num_walkers=1000, num_MC_steps=10000, num_equil_steps=2000)

    # println("📊 Final confirmed energy = $(final_result.mean_energy) ± $(final_result.sem_energy)")

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