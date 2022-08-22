"""
	Sinkhorn algorithm test
	created: 2022, June
	author©: Rajmadan Lakshmanan
"""


using Images, FileIO
using OptimalTransport
using Test
using Tulip

"""
Real data analysis
Data: "DOTmark" dataset, DOTmark:  A Benchmark for Discrete Optimal Transport.
Prominent Sinkhorn algorithms: "OptimalTransport.jl" Julia library 
"""



@testset "2D test:" begin
	printstyled("\n	───── Other Sinkhorn, 2D\n"; color= :green)
	@time emd2(p1, p2, (distFunction(s1,s2)).^rWasserstein,Tulip.Optimizer())
	@time Alg1=sinkhorn(p1, p2, (distFunction(s1,s2)).^rWasserstein, λ)
    @time Alg2=sinkhorn2(p1, p2, (distFunction(s1,s2)).^rWasserstein, λ)
    @time Alg3=sinkhorn_divergence(p1, p2, (distFunction(s1,s2)).^rWasserstein, λ)
    @time Alg4=sinkhorn_stabilized(p1, p2, (distFunction(s1,s2)).^rWasserstein, λ)
    @time Alg5=sinkhorn_stabilized_epsscaling(p1, p2, (distFunction(s1,s2)).^rWasserstein, λ)
	@time NFFT_Sinkhorn_algorithm= SinkhornNFFT2D(p1, p2, s1, s2; rWasserstein= rWasserstein, λ= λ)
end


