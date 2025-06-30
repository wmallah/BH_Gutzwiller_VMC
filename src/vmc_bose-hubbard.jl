# Import packages
using Random
using Distributions
using Statistics
using Optim
using LineSearches

abstract type AbstractLattice end
abstract type Wavefunction end

struct GutzwillerWavefunction{T <: Real} <: Wavefunction
    f::Matrix{T}
end

# Out constructor to allow implict type inference
GutzwillerWavefunction(f::Matrix{<:Real}) = GutzwillerWavefunction{eltype(f)}(f)

struct System{T <: Real}
    t::T
    U::T
    lattice::AbstractLattice
end

struct Lattice
    L::Int
    dims::Tuple{Int,Int}
    BC::String
    adjacency::Vector{Tuple{Int,Int}}
end

# Fix this so that it generates a vector of vectors containing the nearest neighbors for each site
function Lattice1D(L::Int; BC::String="PBC")
    adj = Tuple{Int, Int}[]
    
    for i in 1:L-1
        push!(adj, (i, i+1))
    end

    if BC=="PBC"
        push!(adj, (L,1))
    end

    return Lattice(L, (L,1), BC, adj)
end


function build_neighbor_list_2D(Lx::Int, Ly::Int; periodic::Bool=true)
    L = Lx * Ly
    neighbors = [Int[] for _ in 1:L]

    # Map 2D → 1D: site(x, y) = (y - 1) * Lx + x
    site(x, y) = (mod1(x, Lx) - 1) + (mod1(y, Ly) - 1) * Lx + 1

    for y in 1:Ly
        for x in 1:Lx
            i = site(x, y)

            # Right neighbor
            if periodic || x < Lx
                push!(neighbors[i], site(x + 1, y))
            end

            # Left neighbor
            if periodic || x > 1
                push!(neighbors[i], site(x - 1, y))
            end

            # Up neighbor
            if periodic || y < Ly
                push!(neighbors[i], site(x, y + 1))
            end

            # Down neighbor
            if periodic || y > 1
                push!(neighbors[i], site(x, y - 1))
            end
        end
    end

    return neighbors
end

struct Lattice2D <: AbstractLattice
    Lx::Int
    Ly::Int
    neighbors::Vector{Vector{Int}}
end

function Lattice2D(Lx::Int, Ly::Int; periodic::Bool=true)
    neighbors = build_neighbor_list_2D(Lx, Ly, periodic=periodic)
    return Lattice2D(Lx, Ly, neighbors)
end

function hop_possible(n::Vector{Int}, from::Int, to::Int, f::Matrix{<:Real})
    n_from = n[from]
    n_to = n[to]
    n_max = size(f, 1) - 1
    L = size(f, 2)

    return 1 ≤ from ≤ L && 1 ≤ to ≤ L &&
           0 ≤ n_from ≤ n_max &&
           0 ≤ n_to < n_max &&
           n_from > 0
end

function normalize_wavefunction!(wf::GutzwillerWavefunction)
    norm_squared = sum(wf.f.^2)
    wf.f .= wf.f ./ sqrt(norm_squared)
    return wf
end



function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System)
    f = ψ.f
    lattice = sys.lattice
    t, U = sys.t, sys.U

    # Initialize
    L = length(n)
    E_kin = 0
    E_pot = 0

    # Potential energy (diagonal elements)
    for i in eachindex(n)
        E_pot += (U/2)*n[i]*(n[i]-1)
    end

    # Kinetic energy (off-diagonal elements)
    for i in eachindex(n)
        for j in lattice.neighbors[i]
            if j > i    # ensures each pair (i,j) is visited only once
                if hop_possible(n, j, i, f)
                    R = (f[n[i]+2, i] / f[n[i]+1, i]) * (f[n[j],j] / f[n[j]+1,j])
                    E_kin += -t * sqrt((n[i] + 1) * n[j]) * R
                end

                if hop_possible(n, i, j, f)
                    R = (f[n[j]+2, j] / f[n[j]+1, j]) * (f[n[i],i] / f[n[i]+1,i])
                    E_kin += -t * sqrt((n[j] + 1) * n[i]) * R
                end
            end
        end
    end


    return E_kin + E_pot
end