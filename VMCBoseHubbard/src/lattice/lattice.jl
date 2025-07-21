abstract type AbstractLattice end

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