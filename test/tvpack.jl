using StatsFuns, Test

@testset "bvncdf: bivariate normal cdf" begin

    @test MarkovTransitionMatrices.bvncdf(0.0, 0.0, 0.0) == 0.25
    @test MarkovTransitionMatrices.bvncdf(0.0, Inf, 0.0) == 0.5
    @test MarkovTransitionMatrices.bvncdf(Inf, 0.0, 0.0) == 0.5
    @test MarkovTransitionMatrices.bvncdf(Inf, Inf, 0.0) == 1.0

    for r in -1.0:0.25:1.0
        for x in -10.0:0.5:10.0
            @test MarkovTransitionMatrices.bvncdf(x, Inf, r) == normcdf(x)
            @test MarkovTransitionMatrices.bvncdf(Inf, x, r) == normcdf(x)
            @test MarkovTransitionMatrices.bvncdf(x, -Inf, r) == 0.0
            @test MarkovTransitionMatrices.bvncdf(-Inf, x, r) == 0.0
        end
    end

    @test_throws DomainError MarkovTransitionMatrices.bvncdf(0.0, 0.0, -2.0)
    @test_throws DomainError MarkovTransitionMatrices.bvncdf(0.0, 0.0, 2.0)

    @test MarkovTransitionMatrices.bvncdf(0.0, -100000.0, 0.0) ≈ 0.0
    @test MarkovTransitionMatrices.bvncdf(0.0,  100000.0, 0.0) ≈ 0.5
    @test MarkovTransitionMatrices.bvncdf(100000.0,  100000.0, 0.0000) ≈ 1.0
    @test MarkovTransitionMatrices.bvncdf(-100000.0,  100000.0, 0.0000) ≈ 0.0

    for x in -100.0:100.0, y in -100.0:100.0, r in -1.0:0.1:1.0
        @test !isnan(MarkovTransitionMatrices.bvncdf(x,y,r))
    end
end
