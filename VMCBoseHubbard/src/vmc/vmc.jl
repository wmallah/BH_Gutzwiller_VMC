using Random, Statistics
using ProgressMeter

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
Purpose: perform variational Monte Carlo to integrate the local energy with a fixed number of particles
Input: sys (system struct), κ (variational parameter), n_max (maximum number of particles on a given site), N_total (total number of particles)
Optional Input: num_walkers, num_MC_steps, num_equil_steps
Output: struct of variational Monte Carlo results (see struct defined above)
Author: Will Mallah
Last Updated: 07/16/25
    Summary: Added hot start shuffling, global moves, and changed to generating coefficients outside of loop to avoid unnessecary calculations
=#
function VMC_fixed_particles(sys::System, κ::Real, n_max::Int, N_total::Int;
                             num_walkers::Int=200, num_MC_steps::Int=20000, num_equil_steps::Int=5000)

    # Check that the number of Monte Carlo steps is more than the number of equilibrium steps. Throw error if not.
    if num_MC_steps <= num_equil_steps
        error("num_MC_steps must be greater than num_equil_steps")
    end

    # Intialize number of sites from rows of lattice neigbors matrix
    L = length(sys.lattice.neighbors)

    # Initialize walker values to configurations with fixed number of particles
    walkers = [initialize_fixed_particles(L, N_total, n_max) for _ in 1:num_walkers]

    # 🔥 Pre-shuffle ("hot start") to break symmetry and avoid frozen configs   [IS THIS REALLY NECESSARY?]
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

    # Initialize the energy vectors to fill during measurements
    energies = Float64[]
    kinetic = Float64[]
    potential = Float64[]

    # Keep track of the number of accepted and failed moves for debugging
    num_accepted = 0
    num_failed_moves = 0

    # 
    ψ = generate_coefficients(fill(0, L), κ, n_max)  # Any dummy vector of length L works

    @showprogress enabled=true "Running Canonical VMC..." for step in 1:num_MC_steps
        for i in 1:num_walkers
            n_old = walkers[i]
            n_new = copy(n_old)
            moved = false
            attempt_global = rand() < 0.02  # 2% chance of global hop [MAY REMOVE LATER IF NOT NEEDED]

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

            if step >= num_equil_steps
                E = local_energy(walkers[i], ψ, sys)
                T, V = local_energy_parts(walkers[i], ψ, sys)
                if isfinite(E)
                    push!(energies, E)
                    push!(kinetic, T)
                    push!(potential, V)
                end
            end
        end
    end

    if isempty(energies)
        @warn "No valid energy samples collected! Returning Inf energy."
        println("Bad")
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, 0.0, Float64[], num_failed_moves)
    end

    mean_energy = mean(energies)
    sem_energy = std(energies) / sqrt(length(energies))
    mean_kinetic = mean(kinetic)
    sem_kinetic = std(kinetic) / sqrt(length(kinetic))
    mean_potential = mean(potential)
    sem_potential = std(potential) / sqrt(length(potential))

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

            insert_sites = [i for i in 1:L if n_old[i] < n_max]
            remove_sites = [i for i in 1:L if n_old[i] > 0]

            ΔN = 0

            move_type = rand(Bool)  # true = insert, false = remove

            if move_type && !isempty(insert_sites)
                site = rand(insert_sites)
                n_new[site] += 1
                ΔN = +1

            elseif !move_type && !isempty(remove_sites)
                site = rand(remove_sites)
                n_new[site] -= 1
                ΔN = -1

            else
                # No valid move possible for the chosen type; mark as rejected
                ΔN = 0
            end

            if ΔN == 0
                num_failed_moves += 1
                continue
            else
                r = sampling_ratio(n_old, n_new, κ, n_max) * exp(μ * ΔN)

                if rand() < r
                    walkers[i] = n_new
                    num_accepted += 1
                else
                    num_failed_moves += 1
                    continue
                end
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