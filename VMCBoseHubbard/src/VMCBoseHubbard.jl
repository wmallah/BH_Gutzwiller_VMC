# VMCBoseHubbard.jl

module VMCBoseHubbard

# Include source files — they all define functions/types in *this* module
include("lattice/lattice.jl")
include("system/system.jl")
include("wavefunction/wavefunction.jl")
include("vmc/utils.jl")
include("vmc/moves.jl")
include("vmc/vmc.jl")
include("measurements/measurements.jl")
include("optimizer/KappaOptimizer.jl")

# Load submodules
using .KappaOptimizer: optimize_kappa

# Export user-facing names directly
export Lattice1D, Lattice2D
export System
export GutzwillerWavefunction
export optimize_kappa

end
