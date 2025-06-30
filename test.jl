include("vmc_bose-hubbard.jl")

Lx, Ly = 2, 2                   # Small 2D system (4 sites)
n_max = 3                       # Max occupation per site
n_sites = Lx * Ly

lattice = Lattice2D(Lx, Ly; periodic=true)
system = System(1.0, 2.0, lattice)  # t = 1.0, U = 2.0

f = fill(1.0, n_max + 1, n_sites)  # f[n+1, i] = f_n^{(i)}
ψ = GutzwillerWavefunction(f)

n = fill(1, n_sites)

E = local_energy(n, ψ, system)
println("Local energy: ", E)
