"""
	Sinkhorn approximation
	created: 2022, September
	author©: Rajmadan Lakshmanan
"""

using Test, Distributions

printstyled("\n\t══════════ Test Wasserstein 3D.jl ══════════\n"; bold= true, color= 7)
  

@show n1, n2= 10000, 10000
@testset "3D test:" begin
	printstyled("\n	───── fast Sinkhorn, 3D\n"; color= :green)
	p1= rand(n1); p1/= sum(p1);
	s1= rand(n1, 3)
	p2= rand(n2); p2/= sum(p2)
	s2= rand(n2, 3)
	
	λ= 1.; rWasserstein= 2.;
	#@show A= Wasserstein(p1, p2, distFunction(s1, s2); rWasserstein= rWasserstein)
	@time SS= Sinkhorn(p1, p2, distFunction(s1, s2); rWasserstein= rWasserstein, λ= λ)
	@show SS.distSinkhornUB, SS.count
	@time SF= SinkhornNFFT3D(p1, p2, s1, s2; rWasserstein= rWasserstein, λ= λ)
	@show SF.distSinkhorn, SF.distSinkhornUB, SF.count
	#@test SS.distSinkhorn ≤ A.distance^rWasserstein ≤ SS.distSinkhornUB
	@test SF.distSinkhorn ≈ SS.distSinkhorn atol= 1e-2
end

