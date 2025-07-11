using Pkg
Pkg.activate("..")  # Activate your project environment

include("../src/VMCBoseHubbard.jl")
using .VMCBoseHubbard
using Printf

# System constants
L = 12
Lx, Ly = 2, 2
N_total = 12
_1D = false
t = 1.0
U_vals = 1.0:1.0:10.0

# Output file
outname = "../data/vmc_scan_vs_U.dat"
open(outname, "w") do io
    println(io, "# U     kappa     energy     sem")  # Header

    for U in U_vals
        if _1D == true
            sys = System(t, U, Lattice1D(L))
        else
            sys = System(t, U, Lattice2D(Lx, Ly))
        end
            try
                result = optimize_kappa_grid_scan(sys, N_total)
                @printf(io, "%4.1f  %8.5f  %10.6f  %10.6f\n",
                        U, result.kappa, result.energy, result.sem)
                println("✅ U = $U → κ = $(result.kappa), E = $(result.energy)")
            catch e
                @warn "❌ Failed at U = $U: $e"
            end
    end
end

println("📄 Results written to '$outname'")