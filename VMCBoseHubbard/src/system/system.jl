#=
Purpose: store system parameters and other system information 
Input: t (hopping amplitude), U (interaction strength), μ (chemcial potential), lattice
Author: Will Mallah
Last Updated: 01/25/26
=#
struct System{T <: Real}
    t::T
    U::T
    μ::T
    lattice::AbstractLattice
end