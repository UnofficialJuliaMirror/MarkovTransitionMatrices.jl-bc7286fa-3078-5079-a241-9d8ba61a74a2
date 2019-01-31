# using LinearAlgebra
# using MarkovTransitionMatrices
# using StatsFuns
# using Plots
# using Test

using Base.Iterators
using StatsBase

@testset "tauchen tests" begin

    num_y_nodes = 21
    num_yy_nodes = num_y_nodes^2
    ylim = 3.0

    yspace = range(-abs(ylim); stop=abs(ylim), length=num_y_nodes)
    yyspace = product(yspace,yspace)
    Q = Matrix{Float64}(undef, num_y_nodes, num_y_nodes)
    P = Matrix{Float64}(undef, num_yy_nodes, num_yy_nodes)
    P2 = copy(P)

    μ(x) = x
    Σ = [1.0 0.5; 0.5 1.0]

    Q .= 0.0
    tauchen_1d!(Q, yspace, μ, 1.0)
    @test all(sum(Q, dims=2) .≈ 1.0)
    tauchen_2d!(P, yyspace, μ, Matrix{Float64}(1.0I,2,2))
    @test all(sum(P, dims=2) .≈ 1.0)

    @test P ≈ kron(Q,Q)


    tauchen_2d!(P, yyspace, μ, Σ)
    @test all(sum(P, dims=2) .≈ 1.0)
    P3 = copy(P)
    tauchen_2d!(P, product(yspace.*sqrt2, yspace.*sqrt2), μ, Σ.*2)
    @test P ≈ P3

    @test all(sum(P, dims=2) .≈ 1.0)
    JN, Λ, numMoments, approxErr = bvn_discreteApprox!(P2, yyspace, μ, Σ)
    @test all(sum(P2, dims=2) .≈ 1.0)

    @show countmap(numMoments)

    findmax(abs.(P .- P2))

    # heatmap(P .- P2)
    # heatmap(reshape(P2[41,:] .- P[41,:],21,:))
    # histogram(vec((P2 .- P)[numMoments .== 5,:]), xlim=(-0.005, 0.005))
    # heatmap(reshape(P[73,:],num_y_nodes,num_y_nodes))

    pdiff = [log(x)-log(y) for (x,y) in zip(P,P2) if x > 0 && y > 0]
    # histogram(pdiff)
end
