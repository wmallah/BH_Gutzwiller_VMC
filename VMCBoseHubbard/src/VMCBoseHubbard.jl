
module VMCBoseHubbard

using Random
using Distributions
using ProgressMeter
using Statistics

export Lattice1D, Lattice2D, System, run_vmc, VMCResults, estimate_n_max

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
    if periodic
        neighbors = [Int[] for _ in 1:(Lx * Ly)]
        site(x, y) = (mod1(x, Lx) - 1) + (mod1(y, Ly) - 1) * Lx + 1
        for y in 1:Ly, x in 1:Lx
            i = site(x, y)
            if periodic || x < Lx; push!(neighbors[i], site(x+1, y)); end
            if periodic || x > 1;  push!(neighbors[i], site(x-1, y)); end
            if periodic || y < Ly; push!(neighbors[i], site(x, y+1)); end
            if periodic || y > 1;  push!(neighbors[i], site(x, y-1)); end
        end
        return Lattice2D(Lx, Ly, neighbors)
    else
        error("Open boundary conditions for Lattice2D are not implemented yet.")
    end
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

# Keeps storage dense by always ensuring that f is a Matrix(T)
struct GutzwillerWavefunction{T <: Real} <: Wavefunction
    f::Matrix{T}
end

# Allows acceptance of any AbstractMatrix{<:Real}
GutzwillerWavefunction(f::AbstractMatrix{<:Real}) =
    GutzwillerWavefunction{eltype(f)}(Matrix(f))

#=
Purpose: generate coefficients of the Gutzwiller variational wavefunction
Input: n (vector of integers describing the system state), kappa (variational parameter)
Output: struct of Gutzwiller Wavefunction containing coefficients for a given system state
Author: Will Mallah
Last Updated: 07/04/25
To-Do: 
=#
function generate_coefficients(n::Vector{Int}, κ::Real)
    L = length(n)
    n_max = maximum(n) + 2
    f = zeros(Float64, n_max, L)
    for i in 1:L
        for ni in 0:(n_max - 1)
            f[ni+1, i] = (1 / sqrt(factorial(ni))) * exp(-κ * ni^2 / 2.0)
        end
    end
    return GutzwillerWavefunction(f)
end

#=
Purpose: determine if a hop between sites is possible
Input: n (vector of integers describing the system state), from (site index of hop source), to (site index of hop destination), f (matrix of coefficients from Gutzwiller Wavefunction)
Output: true if hop is possible, false if hop is not possible
Author: Will Mallah
Last Updated: 07/04/25
To-Do: 
=#
function hop_possible(n::Vector{Int}, from::Int, to::Int, f::Matrix{<:Real})
    n_from, n_to = n[from], n[to]
    n_max = size(f, 1) - 1
    L = size(f, 2)
    return 1 ≤ from ≤ L && 1 ≤ to ≤ L &&
           0 ≤ n_from ≤ n_max && 0 ≤ n_to < n_max &&
           n_from > 0
end

#=
Purpose: calculate the local energy
Input: n (vector of integers describing the system state), ψ (wavefunction struct), sys (system struct)
Output: total local energy (kinetic + potential)
Author: Will Mallah
Last Updated: 07/04/25
To-Do: 
=#
function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System)
    f = ψ.f
    lattice = sys.lattice
    t, U = sys.t, sys.U
    L = length(n)
    E_kin, E_pot = 0.0, 0.0
    for i in eachindex(n)
        E_pot += (U/2) * n[i] * (n[i] - 1)
    end
    for i in eachindex(n)
        for j in lattice.neighbors[i]
            if j > i  # visit each pair once
                # hop j → i
                if hop_possible(n, j, i, f)
                    R = f[n[i]+2, i] * f[n[j], j] / (f[n[i]+1, i] * f[n[j]+1, j])
                    E_kin += -t * sqrt((n[i]+1) * n[j]) * R
                end
                # hop i → j
                if hop_possible(n, i, j, f)
                    R = f[n[j]+2, j] * f[n[i], i] / (f[n[j]+1, j] * f[n[i]+1, i])
                    E_kin += -t * sqrt((n[j]+1) * n[i]) * R
                end
            end
        end
    end

    return E_kin + E_pot
end

#=
Purpose: calculate the sampling ratio
Input: n_old (old state of system), n_new (new state of the system), κ (variational parameter)
Output: sampling ratio between two system states
Author: Will Mallah
Last Updated: 07/04/25
To-Do: 
=#
function sampling_ratio(n_old::Vector{Int}, n_new::Vector{Int}, κ::Real)
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
Last Updated: 07/04/25
To-Do: It seems like the code allows hopping between any two random sites but should only allow nearest neighbor hopping. However, this may not matter as were just sampling random system configurations as long as the total number of particles remains the same and the number of particles on any given site does not exceed n_max [CHECK]
=#
function VMC_fixed_particles(sys::System, κ::Real, n_max::Int, N_total::Int;
                             num_walkers::Int=200, num_MC_steps::Int=20000, num_equil_steps::Int=5000)

    if num_MC_steps <= num_equil_steps
    error("num_MC_steps must be greater than num_equil_steps")
    end

    L = length(sys.lattice.neighbors)
    walkers = [initialize_fixed_particles(L, N_total, n_max) for _ in 1:num_walkers]
    energies = zeros(Float64, num_walkers * (num_MC_steps - num_equil_steps))
    idx = 1
    num_accepted = 0
    num_failed_moves = 0

    @showprogress enabled=true "Running VMC..." for step in 1:num_MC_steps
        for i in 1:num_walkers
            n_old = walkers[i]
            n_new = copy(n_old)
            moved = false
            for attempt in 1:20
                # Select random sites to determine if hop is possible
                from, to = rand(1:L, 2)
                # Loop until source and destination sites are different from each other
                while from == to
                    to = rand(1:L)
                end
                # Could replace the stand-alone if statement with the "hop_possible" function
                if n_old[from] > 0 && n_old[to] < n_max
                    n_new[from] -= 1
                    n_new[to] += 1
                    r = sampling_ratio(n_old, n_new, κ)
                    # If the sampling ratio is greater than the random value, accept the move
                    if rand() < r
                        walkers[i] = n_new
                        num_accepted += 1
                    end
                    moved = true
                    break
                end
            end
            if !moved
                num_failed_moves += 1
                continue
            end
            # Make measurement if system equilibrated
            if step > num_equil_steps
                ψ = generate_coefficients(walkers[i], κ)
                E = local_energy(walkers[i], ψ, sys)
                # Only accept finite energies
                if isfinite(E)
                    energies[idx] = E
                    idx += 1
                end
            end
        end
    end

    mean_energy = mean(energies)
    sem_energy = std(energies) / sqrt(length(energies))
    acceptance_ratio = num_accepted / (num_MC_steps * num_walkers)

    return VMCResults(mean_energy, sem_energy, acceptance_ratio, energies, num_failed_moves)
end

function run_vmc(sys::System, κ::Real, n_max::Int; kwargs...)
    lattice = sys.lattice
    if lattice isa Lattice1D
        return VMC_fixed_particles(sys, κ, n_max, lattice.L; kwargs...)
    elseif lattice isa Lattice2D
        return VMC_fixed_particles(sys, κ, n_max, lattice.Lx * lattice.Ly; kwargs...)
    else
        error("Unsupported lattice type: $(typeof(lattice))")
    end
end

# Function to estimate value for n_max given kappa parameter and cutoff value for decay of Gutzwiller coefficients
function estimate_n_max(κ::Real; cutoff::Real = 1e-6)
    n = 0
    while true
        f_n = (1 / sqrt(factorial(n))) * exp(-κ * n^2 / 2)
        if f_n < cutoff
            return n - 1  # previous n was last significant
        end
        n += 1
        if n > 1000
            error("κ too small or cutoff too strict — did not converge")
        end
    end
end

end