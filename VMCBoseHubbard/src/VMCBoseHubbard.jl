
module VMCBoseHubbard

using Random, Statistics, ProgressMeter
using SpecialFunctions: loggamma

# Explicitly export all functions and structs that are called in other files
export Lattice1D, Lattice2D, System, run_vmc, VMCResults, VMC_grand_canonical_adaptive_mu, estimate_n_max, find_mu_for_N_target, tune_mu_and_log

# Create abstract types for both the lattice and wavefunctions so we can 
abstract type AbstractLattice end
abstract type Wavefunction end

# ─── Lattice Types ─────────────────────────────────────────

#=
Purpose: store information about 1D lattice
Input: lattice size, list of neighbors
Author: Will Mallah
Last Updated: 07/04/25
=#
struct Lattice1D <: AbstractLattice
    L::Int
    neighbors::Vector{Vector{Int}}
end


#=
Purpose: generate lattice for 1D system
Input: size of lattice (number of sites), boolean value defaulted to true for periodic boundary conditions
Output: struct containing lattice information
Author: Will Mallah
Last Updated: 07/04/25
To-Do: implement open boundary conditions
=#
function Lattice1D(L::Int; periodic::Bool=true)
    if periodic
        neighbors = [Int[] for _ in 1:L]
        for i in 1:(L-1)
            push!(neighbors[i], i+1)
            push!(neighbors[i+1], i)
        end
        if periodic && L > 2
            push!(neighbors[1], L)
            push!(neighbors[L], 1)
        end
        return Lattice1D(L, neighbors)
    else
       error("Open boundary conditions for Lattice1D are not implemented yet.")
    end
end


#=
Purpose: store information about 2D lattice
Input: Lx, Ly, list of neighbors
Author: Will Mallah
Last Updated: 07/04/25
=#
struct Lattice2D <: AbstractLattice
    Lx::Int
    Ly::Int
    neighbors::Vector{Vector{Int}}
end


#=
Purpose: generate lattice for 2D system
Input: Lx, Ly, boolean value defaulted to true for periodic boundary conditions
Output: struct containing lattice information
Author: Will Mallah
Last Updated: 07/04/25
To-Do: implement open boundary conditions
=#
function Lattice2D(Lx::Int, Ly::Int; periodic::Bool=true)
    neighbors = [Int[] for _ in 1:(Lx * Ly)]

    site(x, y) = (mod1(x, Lx) - 1) + (mod1(y, Ly) - 1) * Lx + 1

    for y in 1:Ly, x in 1:Lx
        i = site(x, y)

        for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1))  # right, left, up, down
            x2, y2 = x + dx, y + dy

            if periodic || (1 ≤ x2 ≤ Lx && 1 ≤ y2 ≤ Ly)
                j = site(x2, y2)
                if j ∉ neighbors[i]
                    push!(neighbors[i], j)
                end
                if i ∉ neighbors[j]
                    push!(neighbors[j], i)
                end
            end
        end
    end

    return Lattice2D(Lx, Ly, neighbors)
end


# ─── System and Wavefunction ───────────────────────────────

#=
Purpose: store system parameter and other system information 
Input: t (hopping amplitude), U (interaction strength), lattice
Author: Will Mallah
Last Updated: 07/04/25
=#
struct System{T <: Real}
    t::T
    U::T
    lattice::AbstractLattice
end


#=
Purpose: generate coefficients of the Gutzwiller variational wavefunction
Input: n (vector of integers describing the system state), kappa (variational parameter), n_max (maximum number of particles on a given site)
Output: struct of Gutzwiller Wavefunction containing coefficients and normalization
Author: Will Mallah
Last Updated: 07/16/25
    Summary: Since coefficient are globally the same, generate them as a vector rather than matrix to reduce number of computations and confusion around indexing
=#
const LOGFACTORIAL_TABLE = [loggamma(m + 1) for m in 0:100]  # supports up to n_max=100

function generate_coefficients(n::Vector{Int}, κ::Real, n_max::Int; logfact=LOGFACTORIAL_TABLE)
    n_cutoff = max(n_max + 2, maximum(n) + 2)   
    f = [exp(-κ * m^2 / 2.0 - 0.5 * logfact[m + 1]) for m in 0:n_cutoff]
    Z = sum(abs2, f)
    # println("Z = $Z")
    return GutzwillerWavefunction(f, Z)
end


struct GutzwillerWavefunction{T <: Real} <: Wavefunction
    f::Vector{T}
    Z::T
end



#=
Purpose: determine if a hop between sites is possible
Input: n (vector of integers describing the system state), from (site index of hop source), to (site index of hop destination), f (matrix of coefficients from Gutzwiller Wavefunction)
Output: true if hop is possible, false if hop is not possible
Author: Will Mallah
Last Updated: 07/04/25
To-Do: 
=#
function hop_possible(n::Vector{Int}, from::Int, to::Int, f::Vector{<:Real})
    L = length(n)
    n_max = length(f) - 1
    n_from = n[from]
    n_to = n[to]

    return 1 ≤ from ≤ L &&
           1 ≤ to ≤ L &&
           n_from > 0 &&
           n_to < n_max
end


#=
Purpose: calculate the local energy
Input: n (vector of integers describing the system state), ψ (wavefunction struct), sys (system struct), n_max (maximum number of particles on a given site)
Output: total local energy (kinetic + potential)
Author: Will Mallah
Last Updated: 07/08/25
    Summary: implemented normalization 
=#
function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System; μ::Real = 0)
    f = ψ.f                     # shared Gutzwiller coefficient vector
    t, U = sys.t, sys.U
    lattice = sys.lattice
    L = length(n)

    E_kin = 0.0
    E_pot = 0.0

    # Potential energy term
    for i in 1:L
        E_pot += (U / 2) * n[i] * (n[i] - 1)    # no need for f-dependent term; cancels in ratio
    end

    # Kinetic energy term
    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j → i
                if hop_possible(n, j, i, f)
                    num = f[n[i]+2] * f[n[j]]
                    den = f[n[i]+1] * f[n[j]+1]
                    R1 = num / den / ψ.Z
                    E_kin += -t * sqrt((n[i]+1) * n[j]) * R1
                end

                # hop i → j
                if hop_possible(n, i, j, f)
                    num = f[n[j]+2] * f[n[i]]
                    den = f[n[j]+1] * f[n[i]+1]
                    R2 = num / den / ψ.Z
                    E_kin += -t * sqrt((n[j]+1) * n[i]) * R2
                end
            end
        end
    end

    # Chemical potential correction
    N = sum(n)

    return E_kin + E_pot #- μ*N    # only add for true grand canonical simulations
end


function local_energy_parts(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System)
    f = ψ.f
    t, U = sys.t, sys.U
    lattice = sys.lattice
    L = length(n)

    E_kin = 0.0
    E_pot = 0.0

    for i in 1:L
        E_pot += (U / 2) * n[i] * (n[i] - 1)
    end

    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                if hop_possible(n, j, i, f)
                    num = f[n[i]+2] * f[n[j]]
                    den = f[n[i]+1] * f[n[j]+1]
                    R1 = num / den / ψ.Z
                    E_kin += -t * sqrt((n[i]+1) * n[j]) * R1
                end
                if hop_possible(n, i, j, f)
                    num = f[n[j]+2] * f[n[i]]
                    den = f[n[j]+1] * f[n[i]+1]
                    R2 = num / den / ψ.Z
                    E_kin += -t * sqrt((n[j]+1) * n[i]) * R2
                end
            end
        end
    end

    return E_kin, E_pot
end


#=
Purpose: calculate the sampling ratio
Input: n_old (old state of system), n_new (new state of the system), κ (variational parameter), n_max (maximum number of particles on a given site)
Output: sampling ratio between two system states
Author: Will Mallah
Last Updated: 07/04/25
=#
function sampling_ratio(n_old::Vector{Int}, n_new::Vector{Int}, κ::Real, n_max::Int)
    ratio = 1.0
    for i in eachindex(n_old)
        if n_old[i] != n_new[i]
            f_old = (1 / sqrt(factorial(n_old[i]))) * exp(-κ * n_old[i]^2 / 2)
            f_new = (1 / sqrt(factorial(n_new[i]))) * exp(-κ * n_new[i]^2 / 2)
            ratio *= abs2(f_new) / abs2(f_old)
        end
    end
    return ratio
end

#=
Purpose: generate initial system states where the particle number on any given site does not exceed n_max
Input: L (system size), N (total number of particles), n_max (maximum number of particles on a given site)
Output: system state where n_i <= n_max
Author: Will Mallah
Last Updated: 07/04/25
To-Do: 
=#
function initialize_fixed_particles(L::Int, N::Int, n_max::Int)
    config = zeros(Int, L)
    for _ in 1:N
        site = rand(1:L)
        while config[site] >= n_max
            site = rand(1:L)
        end
        config[site] += 1
    end
    return config
end


#=
Purpose: store information about the VMC results
Input: mean energy, standard deviation of the mean, acceptance ratio, vector of energies, number of failed moves
Author: Will Mallah
Last Updated: 07/04/25
=#
struct VMCResults
    mean_energy::Float64
    sem_energy::Float64
    mean_kinetic::Float64
    sem_kinetic::Float64
    mean_potential::Float64
    sem_potential::Float64
    acceptance_ratio::Float64
    energies::Vector{Float64}
    num_failed_moves::Int
end


#=
Purpose: perform variational Monte Carlo to integrate the local energy
Input: sys (system struct), κ (variational parameter), n_max (maximum number of particles on a given site), N_total (total number of particles)
Optional Input: num_walkers, num_MC_steps, num_equil_steps
Output: struct of variational Monte Carlo results (see struct defined above)
Author: Will Mallah
Last Updated: 07/16/25
    Summary: Added hot start shuffling, global moves, and changed to generating coefficients outside of loop to avoid unnessecary calculations
=#
function VMC_fixed_particles(sys::System, κ::Real, n_max::Int, N_total::Int;
                             num_walkers::Int=200, num_MC_steps::Int=20000, num_equil_steps::Int=5000)

    if num_MC_steps <= num_equil_steps
        error("num_MC_steps must be greater than num_equil_steps")
    end

    L = length(sys.lattice.neighbors)
    walkers = [initialize_fixed_particles(L, N_total, n_max) for _ in 1:num_walkers]

    # 🔥 Pre-shuffle ("hot start") to break symmetry and avoid frozen configs
    num_hot_sweeps = 1000
    for _ in 1:num_hot_sweeps
        for w in walkers
            from, to = rand(1:L), rand(1:L)
            if from != to && w[from] > 0 && w[to] < n_max
                w[from] -= 1
                w[to] += 1
            end
        end
    end
    # println("🔥 Completed hot start shuffling of walkers")

    energies = zeros(Float64, num_walkers * (num_MC_steps - num_equil_steps))
    kinetic = Float64[]
    potential = Float64[]
    idx = 1
    num_accepted = 0
    num_failed_moves = 0

    ψ = generate_coefficients(fill(0, L), κ, n_max)  # Any dummy vector of length L works

    @showprogress enabled=true "Running VMC..." for step in 1:num_MC_steps
        for i in 1:num_walkers
            n_old = walkers[i]
            n_new = copy(n_old)
            moved = false
            attempt_global = rand() < 0.1  # 10% chance of global hop [MAY REMOVE LATER IF NOT NEEDED]

            sites = shuffle(1:L)

            if attempt_global
                # Global move: allow hops between any two sites [MAY REMOVE LATER IF NOT NEEDED]
                for from in sites
                    if n_old[from] == 0
                        continue
                    end
                    for to in shuffle(1:L)
                        if to == from || n_old[to] >= n_max
                            continue
                        end

                        n_new[from] -= 1
                        n_new[to] += 1
                        r = sampling_ratio(n_old, n_new, κ, n_max)

                        if rand() < r
                            walkers[i] = n_new
                            num_accepted += 1
                        end

                        moved = true
                        break
                    end
                    if moved
                        break
                    end
                end

            else
                # Standard local hop: neighbors only
                for from in sites
                    if n_old[from] == 0
                        continue
                    end
                    for to in shuffle(sys.lattice.neighbors[from])
                        if n_old[to] >= n_max
                            continue
                        end

                        n_new[from] -= 1
                        n_new[to] += 1
                        r = sampling_ratio(n_old, n_new, κ, n_max)

                        if rand() < r
                            walkers[i] = n_new
                            num_accepted += 1
                        end

                        moved = true
                        break
                    end
                    if moved
                        break
                    end
                end
            end

            if !moved
                num_failed_moves += 1
                continue
            end

            if step > num_equil_steps
                E = local_energy(walkers[i], ψ, sys)
                T, V = local_energy_parts(walkers[i], ψ, sys)
                if isfinite(E)
                    energies[idx] = E
                    push!(kinetic, T)
                    push!(potential, V)
                    idx += 1
                end
            end
        end
    end

    if isempty(energies)
        @warn "No valid energy samples collected! Returning Inf energy."
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, 0.0, Float64[], num_failed_moves)
    end

    mean_energy = mean(energies)
    sem_energy = std(energies) / sqrt(length(energies))
    acceptance_ratio = num_accepted / (num_MC_steps * num_walkers)

    return VMCResults(mean_energy, sem_energy, mean_kinetic, sem_kinetic, mean_potential, sem_potential, acceptance_ratio, energies, num_failed_moves)
end


struct GrandCanonicalStats
    mu_trace::Vector{Float64}
    N_trace::Vector{Float64}
    steps::Vector{Int}
end


#=
Purpose: perform variational Monte Carlo to integrate the local energy for the grand canoical
Input: sys (system struct), κ (variational parameter), n_max (maximum number of particles on a given site), N_target (target value for total number of particles), μ (initial guess for the chemical potential)
Optional Input: η (chemical potential tuning rate), update_interval (how many steps between updating chemical potential), num_walkers, num_MC_steps, num_equil_steps
Output: struct of variational Monte Carlo results (see struct defined above)
Author: Will Mallah
Last Updated: 07/16/25
=#
function VMC_grand_canonical_adaptive_mu(sys::System, κ::Real, n_max::Int, 
                                          N_target::Real, μ::Real;
                                          η::Real = 0.1,
                                          μ_bounds::Tuple{Float64, Float64} = (-10.0, 10.0),
                                          update_interval::Int = 500,
                                          num_walkers::Int = 200,
                                          num_MC_steps::Int = 30000,
                                          num_equil_steps::Int = 5000)

    L = length(sys.lattice.neighbors)
    walkers = [zeros(Int, L) for _ in 1:num_walkers]

    # Hot start
    for _ in 1:1000
        for w in walkers
            site = rand(1:L)
            if w[site] < n_max
                w[site] += 1
            end
        end
    end

    ψ = generate_coefficients(fill(0, L), κ, n_max)
    energies = Float64[]
    kinetic = Float64[]
    potential = Float64[]
    total_N = Float64[]
    num_accepted = 0
    num_failed_moves = 0
    acceptance_trace = Float64[]  # Optional: track over time

    mu_trace = Float64[]
    N_trace = Float64[]
    steps = Int[]

    @showprogress enabled=true "Running Grand Canonical VMC with adaptive μ..." for step in 1:num_MC_steps
        for i in 1:num_walkers
            n_old = walkers[i]
            n_new = copy(n_old)

            insert_sites = findall(n_old .< n_max)
            remove_sites = findall(n_old .> 0)

            ΔN = 0

            total_moves = length(insert_sites) + length(remove_sites)
            if total_moves == 0
                continue  # no valid move
            end

            if rand() < length(insert_sites) / total_moves
                site = rand(insert_sites)
                n_new[site] += 1
                ΔN = +1
            else
                site = rand(remove_sites)
                n_new[site] -= 1
                ΔN = -1
            end

            r = sampling_ratio(n_old, n_new, κ, n_max) * exp(μ * ΔN)

            if rand() < r
                walkers[i] = n_new
                num_accepted += 1
            else
                num_failed_moves += 1
                continue
            end

            if step > num_equil_steps
                E = local_energy(walkers[i], ψ, sys; μ=μ)
                T, V = local_energy_parts(walkers[i], ψ, sys)
                if isfinite(E)
                    push!(energies, E)
                    push!(kinetic, T)
                    push!(potential, V)
                    push!(total_N, sum(walkers[i]))
                end
            end
        end

        if step % update_interval == 0 && step > num_equil_steps
            avg_N = mean(total_N)
            Δμ = η * (N_target - avg_N)
            μ += tanh(Δμ) * η  # smooth response

            # Optional clamping
            μ = clamp(μ, μ_bounds[1], μ_bounds[2])

            # Tracking
            push!(mu_trace, μ)
            push!(N_trace, avg_N)
            push!(steps, step)

            empty!(total_N)
        end
        acceptance = num_accepted / (num_accepted + num_failed_moves)
        @debug "Step $step: acceptance ratio = $acceptance"
        push!(acceptance_trace, acceptance)  # optional: store for later plotting
    end

    if isempty(energies)
        @warn "No valid energy samples collected!"
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, 0.0, Float64[], num_failed_moves),
               GrandCanonicalStats(mu_trace, N_trace, steps)
    end

    mean_energy = mean(energies)
    sem_energy = std(energies) / sqrt(length(energies))
    mean_kinetic = mean(kinetic)
    sem_kinetic = std(kinetic) / sqrt(length(kinetic))
    mean_potential = mean(potential)
    sem_potential = std(potential) / sqrt(length(potential))
    acceptance_ratio = num_accepted / (num_MC_steps * num_walkers)

    return VMCResults(mean_energy, sem_energy, mean_kinetic, sem_kinetic, mean_potential, sem_potential, acceptance_ratio, energies, num_failed_moves),
           GrandCanonicalStats(mu_trace, N_trace, steps)
end


# ADD DOCUMENTATION HERE
function find_mu_for_N_target(sys::System, κ::Real, n_max::Int, N_target::Real;
                              η::Float64 = 0.01,
                              μ_init::Float64 = 0.0,
                              tune_steps::Int = 5000,
                              walkers::Int = 200,
                              update_interval::Int = 200)

    _, stats = VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, μ_init;
                                                η = η,
                                                num_walkers = walkers,
                                                num_MC_steps = tune_steps,
                                                num_equil_steps = 0,
                                                update_interval = update_interval)

    return stats.mu_trace[end]  # final tuned value
end


"""
    tune_mu_and_log(sys, κ, n_max, N_target; η=0.01, μ_init=1.0, tune_steps=5000, walkers=200, update_interval=200)

Tunes the chemical potential μ to match N_target for a given κ and logs the μ and ⟨N⟩ evolution.

Returns:
  μ_final::Float64      — final tuned chemical potential
  N_final::Float64      — corresponding particle number
  stats::GrandCanonicalStats  — full tracking info
"""
function tune_mu_and_log(sys::System, κ::Real, n_max::Int, N_target::Real;
                         η::Float64 = 0.01,
                         μ_init::Float64 = 1.0,
                         tune_steps::Int = 5000,
                         walkers::Int = 200,
                         update_interval::Int = 200)

    println("🔧 Autotuning μ for κ = $κ to target N = $N_target")

    _, stats = VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, μ_init;
                                                η = η,
                                                num_walkers = walkers,
                                                num_MC_steps = tune_steps,
                                                num_equil_steps = 0,
                                                update_interval = update_interval)

    # for (s, mu, N) in zip(stats.steps, stats.mu_trace, stats.N_trace)
    #     println("  step = $s, μ = $mu, ⟨N⟩ = $N")
    # end

    μ_final = stats.mu_trace[end]
    N_final = stats.N_trace[end]

    return μ_final, N_final, stats
end


#=
Purpose: run the VMC function for either a 1D or 2D lattice
Input: sys (system struct), κ (variational parameter), n_max (maximum number of particles on a given site), N_total (total number of particles)
Optional Input: num_walkers, num_MC_steps, num_equil_steps (kwargs...)
Output: result from VMC_fixed_particles
Author: Will Mallah
Last Updated: 07/04/25
=#
function run_vmc(sys::System, κ::Real, n_max::Int, N_target::Int; grand_canonical=true, kwargs...)
    lattice = sys.lattice
    if !grand_canonical
        if lattice isa Lattice1D
            return VMC_fixed_particles(sys, κ, n_max, N_target; kwargs...)
        elseif lattice isa Lattice2D
            return VMC_fixed_particles(sys, κ, n_max, N_target; kwargs...)
        else
            error("Unsupported lattice type: $(typeof(lattice))")
        end
    else
        if lattice isa Lattice1D
            return VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, 1.0; kwargs...)
        elseif lattice isa Lattice2D
            return VMC_grand_canonical_adaptive_mu(sys, κ, n_max, N_target, 1.0; kwargs...)
        else
            error("Unsupported lattice type: $(typeof(lattice))")
        end
    end
end


#=
Purpose: estimate value for n_max given kappa parameter and cutoff value for decay of Gutzwiller coefficients
Input: kappa (variational parameter), cutoff (value for coefficient deemed insignificant)
Output: particle number where Gutzwiller coefficients are significant
Author: Will Mallah
Last Updated: 07/04/25
=#
function estimate_n_max(κ::Real; cutoff::Real = 1e-6)
    n = 0
    while true
        f_n = (1 / sqrt(float(factorial(big(n))))) * exp(-κ * n^2 / 2)
        if f_n < cutoff
            return n - 1
        end
        n += 1
        if n > 300  # reduced from 1000
            return 300  # instead of error
        end
    end
end

end     # module