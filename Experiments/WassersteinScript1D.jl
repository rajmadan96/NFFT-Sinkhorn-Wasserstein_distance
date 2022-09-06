"""
	Sinkhorn approximation
	created: 2021, August
	author©: Alois Pichler
"""

using Test, Distributions
include("/HOME1/users/personal/lraj/Downloads/github_NFFT/Wasserstein.jl"); #include("SinkhornFFT.jl")

printstyled("\n\t══════════ Test Wasserstein.jl ══════════\n"; bold= true, color= 7)
samplesFrom= Normal(.5, 1.0)		# Uniform(-1, 1), Normal(0.5, 1.), Exponential(.3)
@show n1= 1000
s1= quantile.(samplesFrom, range(1; length=n1)/ (n1+1))
p1=	Vector{Float64}(undef, n1)	# compute the weights
p1[1]= tmp1= tmp2= cdf(samplesFrom, (s1[1]+s1[2])/2)
for i=2:n1-1
	tmp2= cdf(samplesFrom, (s1[i+1]+s1[i])/2)
	p1[i]= tmp2 - tmp1; tmp1= tmp2
end
p1[n1]= 1- tmp2

@show n2= 1000
s2= rand(samplesFrom, n2); p2= fill(1/n2, n2)

@testset begin
	λ= 1.; rWasserstein= 2.
	#@show A= Wasserstein(p1, p2, distFunction(s1, s2); rWasserstein= rWasserstein)
	@time SS= Sinkhorn(p1, p2, distFunction(s1, s2); rWasserstein= rWasserstein, λ= λ)
	@show "pre-allocation: ", SS.distSinkhorn, SS.distSinkhornUB, SS.count
	@time SF= SinkhornNFFT1D(p1, p2, s1, s2; rWasserstein= rWasserstein, λ= λ)
	#@test SS.distSinkhorn ≤ A.distance^rWasserstein ≤ SS.distSinkhornUB
	@show "Sinkhorn NFFT: ", SF.distSinkhorn, SF.distSinkhornUB, SF.count
	@test SS.distSinkhorn ≈ SF.distSinkhorn atol= 1e-1
end
