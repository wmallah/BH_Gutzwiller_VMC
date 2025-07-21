#=
Purpose: calculate the local energy
Input: n (vector of integers describing the system state), ψ (wavefunction struct), sys (system struct), n_max (maximum number of particles on a given site)
Output: total local energy (kinetic + potential)
Author: Will Mallah
Last Updated: 07/08/25
    Summary: implemented normalization 
=#
function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System; μ::Real = 0)
    f = ψ.f                     # shared Gutzwiller coefficient vector
    t, U = sys.t, sys.U
    lattice = sys.lattice
    L = length(n)

    E_kin = 0.0
    E_pot = 0.0

    # Potential energy term
    for i in 1:L
        E_pot += (U / 2) * n[i] * (n[i] - 1)    # no need for f-dependent terms; cancels in ratio
    end

    # Kinetic energy term
    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                # hop j → i
                if hop_possible(n, j, i, f)
                    num = f[n[i]+2] * f[n[j]]
                    den = f[n[i]+1] * f[n[j]+1]
                    R1 = num / den
                    E_kin += -t * sqrt((n[i]+1) * n[j]) * R1
                end

                # hop i → j
                if hop_possible(n, i, j, f)
                    num = f[n[j]+2] * f[n[i]]
                    den = f[n[j]+1] * f[n[i]+1]
                    R2 = num / den
                    E_kin += -t * sqrt((n[j]+1) * n[i]) * R2
                end
            end
        end
    end

    # Chemical potential correction
    N = sum(n)

    return E_kin + E_pot #- μ*N    # only add for true grand canonical simulations
end


function local_energy_parts(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System)
    f = ψ.f
    t, U = sys.t, sys.U
    lattice = sys.lattice
    L = length(n)

    E_kin = 0.0
    E_pot = 0.0

    for i in 1:L
        E_pot += (U / 2.0) * n[i] * (n[i] - 1)
    end

    for i in 1:L
        for j in lattice.neighbors[i]
            if j > i
                if hop_possible(n, j, i, f)
                    num = f[n[i]+2] * f[n[j]]
                    den = f[n[i]+1] * f[n[j]+1]
                    R1 = num / den
                    E_kin += -t * sqrt((n[i]+1) * n[j]) * R1
                end
                if hop_possible(n, i, j, f)
                    num = f[n[j]+2] * f[n[i]]
                    den = f[n[j]+1] * f[n[i]+1]
                    R2 = num / den
                    E_kin += -t * sqrt((n[j]+1) * n[i]) * R2
                end
            end
        end
    end

    return E_kin, E_pot
end