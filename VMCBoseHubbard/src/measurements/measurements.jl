#=
Purpose: calculate the local energy
Input: n (vector of integers describing the system state), ψ (wavefunction struct), sys (system struct), n_max (maximum number of particles on a given site)
Output: total local energy (kinetic + potential)
Author: Will Mallah
Last Updated: 07/08/25
    Summary: implemented normalization 
=#
function local_energy(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System; μ::Real = 0)
    log_f = ψ.f                     # shared Gutzwiller coefficient vector
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
                if hop_possible(n, j, i, log_f)
                    num = log_f[n[i]+2] + log_f[n[j]]
                    den = log_f[n[i]+1] + log_f[n[j]+1]
                    log_R1 = num - den
                    R1 = exp(log_R1)
                    if !isfinite(R1) || R1 > 1e6
                        @warn "Unphysical ratio R1: $R1 at n[i]=$(n[i]), n[j]=$(n[j])"
                    end
                    E_kin += -t * sqrt((n[i]+1) * n[j]) * R1
                end

                # hop i → j
                if hop_possible(n, i, j, log_f)
                    num = log_f[n[j]+2] + log_f[n[i]]
                    den = log_f[n[j]+1] + log_f[n[i]+1]
                    log_R2 = num - den
                    R2 = exp(log_R2)
                    E_kin += -t * sqrt((n[j]+1) * n[i]) * R2
                end
            end
        end
    end

    # Chemical potential correction
    N = sum(n)

    return E_kin + E_pot # - μ*N    # only add for true grand canonical simulations
end


function local_energy_parts(n::Vector{Int}, ψ::GutzwillerWavefunction, sys::System)
    log_f = ψ.f
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
                if hop_possible(n, j, i, log_f)
                    num = log_f[n[i]+2] + log_f[n[j]]
                    den = log_f[n[i]+1] + log_f[n[j]+1]
                    log_R1 = num - den
                    R1 = exp(log_R1)
                    E_kin += -t * sqrt((n[i]+1) * n[j]) * R1
                end
                if hop_possible(n, i, j, log_f)
                    num = log_f[n[j]+2] + log_f[n[i]]
                    den = log_f[n[j]+1] + log_f[n[i]+1]
                    log_R2 = num - den
                    R2 = exp(log_R2)
                    E_kin += -t * sqrt((n[j]+1) * n[i]) * R2
                end
            end
        end
    end

    return E_kin, E_pot
end


function estimate_energy_gradient(result::VMCResults)
    E_loc = result.energies
    O_κ = result.derivative_log_psi

    if isempty(E_loc) || isempty(O_κ)
        return NaN
    end

    mean_E = mean(E_loc)
    mean_O = mean(O_κ)
    mean_EO = mean(E_loc .* O_κ)

    return 2 * (mean_EO - mean_E * mean_O)
end