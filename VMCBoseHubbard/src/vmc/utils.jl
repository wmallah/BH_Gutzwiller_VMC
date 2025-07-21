using Random

#=
Purpose: generate initial system states where the particle number on any given site does not exceed n_max
Input: L (system size), N (total number of particles), n_max (maximum number of particles on a given site)
Output: system state where n_i <= n_max
Author: Will Mallah
Last Updated: 07/04/25
To-Do: 
=#
function initialize_fixed_particles(L::Int, N::Int, n_max::Int)
    # Create vector of length L with integer values to represent system state
    config = zeros(Int, L)

    # Loop through the total number of particles in the system to randomly fill the configuration
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