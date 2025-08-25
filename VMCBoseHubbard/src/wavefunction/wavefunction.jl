using SpecialFunctions: loggamma
using LogExpFunctions: logsumexp

abstract type Wavefunction end

#=
Purpose: generate coefficients of the Gutzwiller variational wavefunction
Input: n (vector of integers describing the system state), kappa (variational parameter), n_max (maximum number of particles on a given site)
Output: struct of Gutzwiller Wavefunction containing coefficients and normalization
Author: Will Mallah
Last Updated: 07/16/25
    Summary: Since coefficient are globally the same, generate them as a vector rather than matrix to reduce number of computations and confusion around indexing
=#

# Precompute factorial values to avoid redundant calculations
const LOGFACTORIAL_TABLE = [loggamma(m + 1) for m in 0:100]  # supports up to n_max=100

function generate_coefficients(κ::Real, n_max::Int; logfact=LOGFACTORIAL_TABLE)
    # Only compute 1 coefficient past the maximum number of particles allowed per site because kinetic energy computes f_n+1
    n_cutoff = n_max + 2
    log_f = [-κ * m^2 / 2.0 - 0.5 * logfact[m + 1] for m in 0:n_cutoff]
    # Normalize the coefficients
    log_Z = logsumexp(2 .* log_f)
    f_normalized = log_f .- 0.5 * log_Z
    return GutzwillerWavefunction(f_normalized)
end


struct GutzwillerWavefunction{T <: Real} <: Wavefunction
    # Vector for coefficients
    f::Vector{T}
end