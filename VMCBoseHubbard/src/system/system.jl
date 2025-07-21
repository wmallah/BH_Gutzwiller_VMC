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