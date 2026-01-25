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
    derivative_log_psi::Vector{Float64}
    num_failed_moves::Int
    PN::Vector{Int}
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

    # Generate the coefficients for the wavefunction
    ψ = generate_coefficients(κ, n_max)  # Any dummy vector of length L works

    @showprogress enabled=true "Running Canonical VMC..." for step in 1:num_MC_steps
        for i in 1:num_walkers
            n_old = walkers[i]
            n_new = copy(n_old)
            moved = false

            sites = shuffle(1:L)

            # Standard local hop: neighbors only
            for from in sites
                # If the source site has zero particles on it, choose a different source site
                if n_old[from] == 0
                    continue
                end
                for to in shuffle(sys.lattice.neighbors[from])
                    # If the destination neighbor site has n_max or more particles on it, randomly choose a different neighbor destination site
                    if n_old[to] >= n_max
                        continue
                    end

                    # If move is valid, update new configuration
                    n_new[from] -= 1
                    n_new[to] += 1

                    # Calculate the sampling ratio for the original and update configurations (probability of accepting move)
                    r = sampling_ratio(n_old, n_new, κ, n_max)

                    # If the randomly generated value [0,1) is less than the sampling ration, accept the move
                    if rand() < r
                        # Add accepted configuration to set of walkers
                        walkers[i] = n_new
                        num_accepted += 1
                    end

                    # Regardless of acceptance status, break out of loop
                    moved = true
                    break
                end
                # Once again, regardless of acceptance status, break out of inner loop to re-enter outer loop, choosing a new source site
                if moved
                    break
                end
            end

            # If there were no possible moves to make, track failed move attempts
            if !moved
                num_failed_moves += 1
            end

            # After equilibration, begin making measurements
            if step >= num_equil_steps
                E = local_energy(walkers[i], ψ, sys)
                if E > 1e4
                    @warn "High energy"
                end
                T, V = local_energy_parts(walkers[i], ψ, sys)
                # Only keep measurements with finite values
                if isfinite(E)
                    push!(energies, E)
                    push!(kinetic, T)
                    push!(potential, V)
                end
            end
        end
    end

    # If no measurements (no possible moves) were made, return Inf for all measurements values, effectively detering optimizer from this region of parameter space
    if isempty(energies)
        @warn "No valid energy samples collected! Returning Inf energy."
        println("Bad")
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, 0.0, Float64[], num_failed_moves)
    end

    # If the energies vector is not empty, calculate the mean and standard deviations of all energy types as well as the move acceptance ratio
    mean_energy = mean(energies)
    sem_energy = std(energies) / sqrt(length(energies))
    mean_kinetic = mean(kinetic)
    sem_kinetic = std(kinetic) / sqrt(length(kinetic))
    mean_potential = mean(potential)
    sem_potential = std(potential) / sqrt(length(potential))

    acceptance_ratio = num_accepted / (num_MC_steps * num_walkers)

    return VMCResults(mean_energy, sem_energy, mean_kinetic, sem_kinetic, mean_potential, sem_potential, acceptance_ratio, energies, num_failed_moves)
end


# Function to ensure our walkers (system configurations) have physical entries
function check_and_warn_walker(n::Vector{Int}, n_max::Int)
    if any(isnan, n)
        @warn "Walker contains NaN: $n"
        return false
    elseif any(isinf, n)
        @warn "Walker contains Inf: $n"
        return false
    elseif any(x -> x < 0 || x > n_max, n)
        @warn "Walker has out-of-bounds occupation: $n"
        return false
    end
    return true
end

#=
Purpose: perform variational Monte Carlo to integrate the local energy for the grand canoical
Input: sys (system struct), κ (variational parameter), n_max (maximum number of particles on a given site), N_target (target value for total number of particles), μ (initial guess for the chemical potential)
Optional Input: num_walkers, num_MC_steps, num_equil_steps
Output: struct of variational Monte Carlo results (see struct defined above)
Author: Will Mallah
Last Updated: 07/16/25
=#
function VMC(sys::System, N_target::Int, κ::Real, n_max::Int, μ::Real, grand_canonical, projective;
                              num_walkers::Int = 200,
                              num_MC_steps::Int = 30000,
                              num_equil_steps::Int = 5000)

    # Extract the system size from the number of rows in the adjacency matrix
    L = length(sys.lattice.neighbors)

    # Function that takes in the system size and target number of particles and returns a random array for the system configuration
    function random_walker(L::Int, N::Int)
        idx = randperm(L)[1:N]
        w = zeros(Int, L)
        for i in idx
            w[i] = 1
        end
        return w
    end

    # Generate an array of random walkers (system configurations)
    walkers = [random_walker(L, N_target) for _ in 1:num_walkers]    

    # Generate the coefficients for the Gutzwiller wavefunction
    wf = generate_coefficients(κ, n_max)

    # Histogram and statistics
    PN = zeros(Int, 2000)
    num_accepted_moves, num_failed_moves = 0, 0

    # Create empty arrays for all the measurements we want to track
    energies, derivative_log_psi, kinetic, potential, total_N = Float64[], Float64[], Float64[], Float64[], Float64[]

    # Begin Monte Carlo Loop outer loop (number of steps in our simulation)
    @showprogress enabled=true "Running Grand Canonical VMC..." for step in 1:num_MC_steps
        # Begin Monte Carlo inner loop (number of walkers/configurations)
        for i in 1:num_walkers
            # Initialize the old and new sets of configurations
            n_old = walkers[i]
            n_new = copy(n_old)

            # Randomly select a site of the current configuration
            site = rand(1:L)

            if grand_canonical
                # Add or remove a particle on this site with 50/50 probability
                if rand() < 0.5
                    n_new[site] += 1
                else
                    n_new[site] -= 1
                end
            else
                # Hop particle from random site to random neigbor site
                from = site
                to = rand(sys.lattice.neighbors[from])

                n_new[from] -= 1
                n_new[to] += 1

                if n_new[to] > n_max
                    num_failed_moves += 1
                    continue
                end
            end

            # Reject proposed move if unphysical
            if n_new[site] > n_max || n_new[site] < 0
                num_failed_moves += 1
                continue
            else
                # Single-site Gutzwiller log acceptance ratio:
                # log( |Ψ(new)|^2 / |Ψ(old)|^2 ) = 2*(log|f(n_new)| - log|f(n_old)|)
                n0 = n_old[site]
                n1 = n_new[site]
                log_ratio = 2.0 * (wf.f[n1 + 1] - wf.f[n0 + 1])

                # Accept move based on Metropolis-Hastings
                if isfinite(log_ratio) && log(rand()) < log_ratio
                    walkers[i] = n_new
                    num_accepted_moves += 1
                else
                    num_failed_moves += 1
                    continue
                end
            end
            
            # Histogram the total number of particles from the configurations
            N_now = sum(walkers[i])
            if N_now + 1 <= length(PN)
                PN[N_now + 1] += 1
            end

            # Check to make sure the walkers have physically correct entries
            if check_and_warn_walker(walkers[i], n_max)
                # Only make measurements after equilibration and with the target number of particles in the system
                if step >= num_equil_steps
                    # If we are not projecting (non-projective grand canonical or canonical), measure. If we are projecting (projective grand canonical), only measure if the number of particles is our target number of particles
                    if !projective || (projective && N_now == N_target)
                        # Measure the total local energy as well as the kinetic and potential energies separately
                        E, T, V = local_energy(walkers[i], wf, sys; μ=μ, n_max=n_max)

                        # If the energy energy is finite, push to the respective vectors
                        if isfinite(E)
                            push!(energies, E)
                            push!(kinetic, T)
                            push!(potential, V)
                            push!(total_N, N_now)
                            # Calculate and track this value (derivative of log psi) for the gradient in our Gradient Descent optimization method
                            val = -0.5 * sum(walkers[i] .^ 2)
                            if !isfinite(val)
                                @warn "Non-finite derivative_log_psi value: $val"
                            else
                                push!(derivative_log_psi, val)
                            end
                        else
                            @warn "Non-finite local energy detected: E = $E"
                            continue
                        end
                    end
                end
            else
                @warn "Invalid walker skipped"
            end
        end
    end

    # Calculate the acceptance ratio to check if the simulation is accepting or rejecting most proposed moves
    acceptance_ratio = num_accepted_moves / (num_accepted_moves + num_failed_moves)

    if isempty(energies)
        @warn "No valid energy samples collected!"
        return VMCResults(Inf, Inf, Inf, Inf, Inf, Inf, 0.0, Float64[], Float64[], num_failed_moves, Int[])
    end

    return VMCResults(
        mean(energies), std(energies) / sqrt(length(energies)),
        mean(kinetic), std(kinetic) / sqrt(length(kinetic)),
        mean(potential), std(potential) / sqrt(length(potential)),
        acceptance_ratio, energies, derivative_log_psi, num_failed_moves, PN
    )
end