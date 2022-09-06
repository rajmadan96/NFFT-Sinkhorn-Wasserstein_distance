"""
	Sinkhorn approximation
	created: 2021, May
	author©: Alois Pichler
"""

using Test
using OptimalTransport
using Tulip
using Clp
#include("Wasserstein.jl")
#include("SinkhornFFT.jl")
#include("SinkhornFFT2D.jl")

printstyled("\n\t══════════ Test Wasserstein.jl ══════════\n"; bold= true, color= 7)


rWasserstein= 2.
@show n1, n2= 1000, 1000
@testset "2D test:" begin
	printstyled("\n	───── fast Sinkhorn, 2D\n"; color= :green)
	p1= rand(n1); p1/= sum(p1);
	s1= rand(n1, 2)
	p2= rand(n2); p2/= sum(p2)
	s2= rand(n2, 2)
	
	λ= 1.; rWasserstein= 2.; ε = 0.1
	@show A= Wasserstein(p1, p2, distFunction(s1, s2); rWasserstein= rWasserstein)
	@time SS= Sinkhorn(p1, p2, distFunction(s1, s2); rWasserstein= rWasserstein, λ= λ)
	@show SS.distSinkhornUB, SS.count
	@time SF= SinkhornNFFT2D(p1, p2, s1, s2; rWasserstein= rWasserstein, λ= λ)
	@show SF.distSinkhorn, SF.distSinkhornUB, SF.count
	@test SS.distSinkhorn ≤ A.distance^rWasserstein ≤ SS.distSinkhornUB
	@test SF.distSinkhorn ≈ SS.distSinkhorn atol= 1e-2
end
