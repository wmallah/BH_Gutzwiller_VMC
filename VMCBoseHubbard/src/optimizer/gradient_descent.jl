import ..VMCBoseHubbard: VMC_grand_canonical
import ..VMCBoseHubbard: VMC_fixed_particles
import ..VMCBoseHubbard: estimate_energy_gradient

export optimize_kappa

function optimize_kappa(sys::System, n_max::Int, μ::Float64;
                          κ_init::Float64 = 1.0,
                          η::Float64 = 0.05,
                          N_target::Int = 12,
                          num_iters::Int = 40,
                          num_walkers::Int = 200,
                          num_MC_steps::Int = 2000,
                          num_equil_steps::Int = 500)

    κ = κ_init
    history = []

    for iter in 1:num_iters
        println("[Iteration $iter] κ = $(round(κ, digits=12))")

        result = VMC_grand_canonical(sys, κ, n_max;
                                    μ = μ,
                                    N_target=N_target,
                                    num_walkers=num_walkers,
                                    num_MC_steps=num_MC_steps,
                                    num_equil_steps=num_equil_steps,
                                    track_derivative=true)  # <-- important!

        # Collect energy and gradient data
        E = result.mean_energy
        grad = estimate_energy_gradient(result)  # Use O_κ field

        println("    Energy = $(round(E, digits=6)), Gradient = $(round(grad, digits=6))")

        # Update
        κ -= η * grad
        κ = clamp(κ, 1e-12, 10.0)

        if !isfinite(κ) || !isfinite(grad)
            @warn "Stopping: Non-finite κ or gradient"
            break
        end

        push!(history, (κ=κ, energy=E, gradient=grad))
    end

    # Return best κ found
    best = findmin(h -> h.energy, history)[2]
    return history[best].κ, history
end