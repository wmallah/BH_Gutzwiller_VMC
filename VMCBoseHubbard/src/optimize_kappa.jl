using Optim
using Statistics
include("VMCBoseHubbard.jl")
using .VMCBoseHubbard

# ─── Choose Dimension and Parameters ────────────────────────────

dim = :_1D                      # use :_1D or :_2D
L = 12                          # for 1D: number of sites
Lx, Ly = 3, 3                   # for 2D: grid dimensions
N_total = 12                    # total number of bosons
t = 1.0                         # hopping
U = 3.3578                      # on-site interaction

# ─── Create Lattice and System ──────────────────────────────────

if dim == :_1D
    lattice = Lattice1D(L)
elseif dim == :_2D
    lattice = Lattice2D(Lx, Ly)
else
    error("Invalid dimension: use :_1D or :_2D")
end

sys = System(t, U, lattice)

# ─── Logging Arrays ─────────────────────────────────────────────

kappas = Float64[]
energies = Float64[]
rejected_kappas = Float64[]
rejected_nmax = Int[]

# ─── Energy Evaluation with Autoscaling on n_max ────────────────

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

        result = run_vmc(sys, κ, n_max; kwargs...)
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

# ─── Optimization Setup ─────────────────────────────────────────

opt_options = Optim.Options(iterations = 20, show_trace = true)
using Random
Random.seed!(42)  # reproducibility

# ─── Stage 1: Global Simulated Annealing ───────────────────────

println("🌍 Starting global optimization (Simulated Annealing)...")
global_result = optimize(x -> energy_for_kappa_logged(x, sys;
                                                      num_walkers = 100,
                                                      num_MC_steps = 10000,
                                                      num_equil_steps = 2000),
                         [1.0],
                         SimulatedAnnealing(),
                         opt_options)

global_kappa = global_result.minimizer[1]
println("🌍 SA result: κ = $global_kappa, E = $(global_result.minimum)")

# ─── Stage 2: Local Refinement (Fminbox + NelderMead) ──────────

println("\n🔍 Starting local refinement from SA result...")
lower_bound = 0.5
upper_bound = 2.0
initial_kappa = clamp(global_kappa, lower_bound, upper_bound)

local_result = optimize(x -> energy_for_kappa_logged(x, sys;
                                                     num_walkers = 100,
                                                     num_MC_steps = 10000,
                                                     num_equil_steps = 2000),
                        [lower_bound], [upper_bound],
                        [initial_kappa],
                        Fminbox(NelderMead()),
                        Optim.Options(iterations = 10, show_trace = true))


refined_kappa = local_result.minimizer[1]
refined_energy = local_result.minimum
println("🔍 Refined result: κ = $refined_kappa")
println("🏁 Refined energy = $refined_energy")

# ─── Final High-Precision Evaluation ───────────────────────────

n_max_final = estimate_n_max(refined_kappa)
final_result = run_vmc(sys, refined_kappa, n_max_final;
                       num_walkers = 1000,
                       num_MC_steps = 50000,
                       num_equil_steps = 10000)

println("📊 Final confirmed energy = $(final_result.mean_energy) ± $(final_result.sem_energy)")

# ─── Show Rejected κ Values (Optional) ─────────────────────────

if !isempty(rejected_kappas)
    println("\n⚠️  Rejected κ values:")
    for (k, n) in zip(rejected_kappas, rejected_nmax)
        println("  κ = $k → estimate_n_max = $n")
    end
end
