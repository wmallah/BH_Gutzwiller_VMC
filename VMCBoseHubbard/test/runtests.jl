using Test
using VMCBoseHubbard

@testset "Basic VMC Tests" begin

    @testset "1D VMC test" begin
        lattice1D = Lattice1D(4)
        sys1D = System(1.0, 2.0, lattice1D)
        result1D = run_vmc(sys1D, 0.3, 4;
                           num_walkers=10,
                           num_MC_steps=100,
                           num_equil_steps=20)

        @test isfinite(result1D.mean_energy)
        @test result1D.acceptance_ratio > 0
        @test length(result1D.energies) == 10 * (100 - 20)
    end

    @testset "2D VMC test" begin
        lattice2D = Lattice2D(2, 2)
        sys2D = System(1.0, 2.0, lattice2D)
        result2D = run_vmc(sys2D, 0.3, 4;
                           num_walkers=10,
                           num_MC_steps=100,
                           num_equil_steps=20)

        @test isfinite(result2D.mean_energy)
        @test result2D.acceptance_ratio > 0
        @test length(result2D.energies) == 10 * (100 - 20)
    end

end
