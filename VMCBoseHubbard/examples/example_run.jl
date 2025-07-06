using Pkg
Pkg.activate("../")              # Activate the project environment in the parent directory

using VMCBoseHubbard

kappa = 1.4
n_max = estimate_n_max(kappa)

function compare_1D_2D_energies(; 
        L1D=12, 
        Lx=4, Ly=4, 
        t=1.0, U=3.3578, 
        κ=kappa, 
        n_max = n_max, 
        N_total_1D=12, 
        N_total_2D=16, 
        num_walkers=1000, 
        num_MC_steps=10000, 
        num_equil_steps=2000)

    # 1D simulation
    println("Running 1D VMC...")
    lattice1D = Lattice1D(L1D)
    sys1D = System(t, U, lattice1D)
    res1D = run_vmc(sys1D, κ, n_max; 
                    num_walkers=num_walkers, 
                    num_MC_steps=num_MC_steps, 
                    num_equil_steps=num_equil_steps)

    println("1D energy = $(res1D.mean_energy) ± $(res1D.sem_energy)")

    # 2D simulation
    println("Running 2D VMC...")
    lattice2D = Lattice2D(Lx, Ly)
    sys2D = System(t, U, lattice2D)
    res2D = run_vmc(sys2D, κ, n_max; 
                    num_walkers=num_walkers, 
                    num_MC_steps=num_MC_steps, 
                    num_equil_steps=num_equil_steps)

    println("2D energy = $(res2D.mean_energy) ± $(res2D.sem_energy)")

    return res1D, res2D
end

# Call the function
compare_1D_2D_energies()
