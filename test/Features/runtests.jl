using Test
using Distributions
using StableRNGs
using StatsBase
using LinearAlgebra
using Random

using RandomFeatures.Features

@testset "Features" begin
   @testset "ScalarFunctions" begin

       af_list = [
           Relu(),
           Lrelu(),
           Gelu(),
           Elu(),
           Selu(),
           Heaviside(),
           SmoothHeaviside(),
           Sawtooth(),
           Softplus(),
           Tansig(),
           Sigmoid(),
       ]
       # very rough tests that these are activation functions
       for af in af_list
           @test isa(af,ScalarActivation)

           x_test_neg = collect(-1:0.1:-0.1)
           x_test_pos = collect(0:0.1:1)
           println("Testing", af)
           @test all(apply_scalar_function(af, x_test_neg) .<= log(2)) # small for negative x
           
           if !isa(af,Sawtooth)
               @test all(apply_scalar_function(af, x_test_pos[2:end]) - apply_scalar_function(af, x_test_pos[1:end-1]) .>= 0) # monotone increasing for positive x
           else
               x_test_0_0pt5 = collect(0:0.1:0.5)
               x_test_0pt5_1 = collect(0.5:0.1:1)
               @test all(apply_scalar_function(af, x_test_0_0pt5[2:end]) - apply_scalar_function(af, x_test_0_0pt5[1:end-1]) .>= 0) 
               @test all(apply_scalar_function(af, x_test_0pt5_1[2:end]) - apply_scalar_function(af, x_test_0pt5_1[1:end-1]) .<= 0) 
           end
       end
       
        

    end
    


    
end
