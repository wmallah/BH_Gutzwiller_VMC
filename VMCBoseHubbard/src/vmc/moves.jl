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