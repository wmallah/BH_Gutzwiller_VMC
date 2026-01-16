import ..VMCBoseHubbard: VMC_grand_canonical
import ..VMCBoseHubbard: VMC_fixed_particles
import ..VMCBoseHubbard: estimate_energy_gradient

export optimize_kappa

function optimize_kappa(sys::System, n_max::Int, μ::Float64;
                          κ_init::Float64 = 1.0,
                          η::Float64 = 0.05,
                          N_target::Int = 2,
                          num_walkers::Int = 200,
                          num_MC_steps::Int = 2000,
                          num_equil_steps::Int = 500)

    κ = κ_init
    history = []

    # ---- Initial evaluation ----
    result_old = VMC_grand_canonical(
        sys, κ, n_max, μ, N_target;
        num_walkers = num_walkers,
        num_MC_steps = num_MC_steps,
        num_equil_steps = num_equil_steps,
        track_derivative = true
    )

    E_old   = result_old.mean_energy
    err_old = result_old.sem_energy
    grad    = estimate_energy_gradient(result_old)

    push!(history, (κ = κ, energy = E_old, gradient = grad))

    # ---- First mandatory update ----
    κ -= η * grad
    κ = clamp(κ, 1e-12, 10.0)

    while true
        result_new = VMC_grand_canonical(
            sys, κ, n_max, μ, N_target;
            num_walkers = num_walkers,
            num_MC_steps = num_MC_steps,
            num_equil_steps = num_equil_steps,
            track_derivative = true
        )

        E_new   = result_new.mean_energy
        err_new = result_new.sem_energy
        grad    = estimate_energy_gradient(result_new)

        println("κ = $(round(κ, digits=15))  E = $(round(E_new, digits=8)) ± $(round(err_new, digits=8))")

        push!(history, (κ = κ, energy = E_new, gradient = grad))

        # ---- Statistical stopping condition ----
        if abs(E_new - E_old) < err_new
            break
        end

        # ---- Gradient descent update ----
        κ -= η * grad
        κ = clamp(κ, 1e-12, 10.0)

        if !isfinite(κ) || !isfinite(grad)
            @warn "Stopping: non-finite κ or gradient"
            break
        end

        E_old = E_new
    end

    _, best_idx = findmin(h -> h.energy, history)
    return history[best_idx].κ, history
end
