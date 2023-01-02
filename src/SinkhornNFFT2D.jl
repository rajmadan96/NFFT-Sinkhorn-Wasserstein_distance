"""
	NFFT accelerated Sinkhorn 2D
	created: 2021, November
	author©: Alois Pichler, Rajmadan Lakshmanan
"""

using NFFT3
using LinearAlgebra
#include("NFFT/NFCT.jl")
#include("NFFT/NFFT.jl")
#include("NFFT/NFFT3.jl")
#include("NFFT/NFST.jl")



"""
	SinkhornNFFT1D(
        p1, p2, s1,s2,rWasserstein, λ)

Construct an NFFT (2D) Sinkhorn algorithm for computing an entropically regularized Wasserstein distance.
"""



#	│	Sinkhorn-Knopp iteration algorithm (2D), NFFT
#	╰────────────────────────────────────────────────────
function SinkhornNFFT2D(p1::Vector{Float64}, p2::Vector{Float64}, S1, S2; rWasserstein::Float64= 1., λ::Float64= 1.,tol=1e-5,max_iter=1000,verbose=false)
	#	│	prepare NFFT and create a Julia plan-object
	#	╰────────────────────────────────────────────────────
	if length(S1)==2*length(p1) && length(S2)==2*length(p2)		# two dimensions
		p= 8; m= p; n= 256; flags= 0
		eps_I= p/ n; eps_B= max(1/ 16, p/ n); nn= 2 * n
		reScale= (0.25- eps_B/ 2)/ max(1, maximum(norm(s) for s in eachrow(S1)), maximum(norm(s) for s in eachrow(S2)))
		if rWasserstein == 1
			kernel= "laplacian_rbf"; kScale= reScale/ λ				#	exp(-λ d)
		elseif rWasserstein == 2
			kernel= "gaussian";		 kScale= reScale/ sqrt(λ)		#	exp(-λ d^2)
		end
		plan1 = NFFT3.FASTSUM(2, size(S2,1), size(S1,1), n, p, kernel, kScale, eps_I, eps_B, nn, m)
		plan2 = NFFT3.FASTSUM(2, size(S1,1), size(S2,1), n, p, kernel, kScale, eps_I, eps_B, nn, m)
		plan1.x = S2* reScale; plan1.y = S1* reScale
		plan2.x = S1* reScale; plan2.y = S2* reScale
	end

	#	│	iterate Sinkhorn
	#	╰────────────────────────────────────────────────────
	ontoSimplex!(p1); βr= Vector{ComplexF64}(undef, length(p1))
	ontoSimplex!(p2); γc= ones(ComplexF64, length(p2))		# guess a starting value
	count= 0; tmpS= Vector{Float64}(undef, length(p1)); 
	while true  # Sinkhorn iteration
		γc= γc./ (p2'*γc)		# rescale
		@. γc= complex(real(γc), 0.)
		tmp= reinterpret(Float64, γc)
		tmp[2:2:end].= 0.0		# force imaginary part zero
		plan1.alpha = γc;
		NFFT3.trafo(plan1)		# fast summation
		plan2.alpha= βr= p1./ plan1.f		# vector operation
		NFFT3.trafo(plan2)
		tmpS= plan2.f
		γc= p2./ tmpS			# vector operation
		count += 1; 
		if count % 10 == 0
		err_βr=
			 norm(real((plan1.f).* βr)- real(p1),2) #/ max(maximum(abs.(real(βr))), maximum(abs.(real(βr_old))), 1)
		 err_γc=
			 norm(real((plan2.f).*γc)- real(p2),2) #/ max(maximum(abs.(real(γc))), maximum(abs.(real(γc_old))), 1)
		 if verbose
			 println("Iteration $count, err = ",  (err_βr+ err_γc))
		 end
		 if ((err_βr+ err_γc) < tol) || count > max_iter
			 break
		 end
		end 
	 
	end 
	tmp=1+p1'*replace!(log.(βr),-Inf=>0) + p2'*replace!(log.(γc),-Inf=>0)- tmpS'* γc #replaced -Inf with 0 to avoid NaN when computing p1'*log.(βr) and p2'*log.(γc)
	return (distSinkhorn= tmp/ λ, distSinkhornUB= (tmp-p1'*replace!(log.(p1),-Inf=>0)-p2'*replace!(log.(p2),-Inf=>0))/ λ, count= count)

end
	
function log_SinkhornNFFT2D(p1::Vector{Float64}, p2::Vector{Float64}, S1, S2; rWasserstein::Float64= 1., λ::Float64= 1.,tol=1e-5,max_iter=1000,verbose=false)
	#	│	prepare NFFT and create a Julia plan-object
	#	╰────────────────────────────────────────────────────
	if length(S1)==2*length(p1) && length(S2)==2*length(p2)		# two dimensions
		p= 8; m= p; n= 256; flags= 0
		eps_I= p/ n; eps_B= max(1/ 16, p/ n); nn= 2 * n
		reScale= (0.25- eps_B/ 2)/ max(1, maximum(norm(s) for s in eachrow(S1)), maximum(norm(s) for s in eachrow(S2)))
		if rWasserstein == 1
			kernel= "laplacian_rbf"; kScale= reScale/ λ				#	exp(-λ d)
		elseif rWasserstein == 2
			kernel= "gaussian";		 kScale= reScale/ sqrt(λ)		#	exp(-λ d^2)
		end
		plan1 = NFFT3.FASTSUM(2, size(S2,1), size(S1,1), n, p, kernel, kScale, eps_I, eps_B, nn, m)
		plan2 = NFFT3.FASTSUM(2, size(S1,1), size(S2,1), n, p, kernel, kScale, eps_I, eps_B, nn, m)
		plan1.x = S2* reScale; plan1.y = S1* reScale
		plan2.x = S1* reScale; plan2.y = S2* reScale
	end

	#	│	iterate Sinkhorn
	#	╰────────────────────────────────────────────────────
	ontoSimplex!(p1);βr= zeros(ComplexF64, length(p1))
	ontoSimplex!(p2); γc= zeros(ComplexF64, length(p2))		# guess a starting value
	count= 0; tmpS= Vector{Float64}(undef, length(p1)); 
	while true  # Sinkhorn iteration
		γc= exp.(λ*γc)		# rescale
		@. γc= complex(real(γc), 0.)
		tmp= reinterpret(Float64, γc)
		tmp[2:2:end].= 0.0		# force imaginary part zero
		plan1.alpha = γc;
		NFFT3.trafo(plan1)		# fast summation
		βr= (log.(p1).- log.(plan1.f))/λ
		plan2.alpha= exp.(λ*βr)		# vector operation
		NFFT3.trafo(plan2)
		tmpS= plan2.f
		γc= (log.(p2).- log.(tmpS))/λ			# vector operation		
		count += 1; 
		if count % 10 == 0
		err_βr=
			 norm(real((plan1.f).* exp.(λβr))- real(p1),2) #/ max(maximum(abs.(real(βr))), maximum(abs.(real(βr_old))), 1)
		 err_γc=
			 norm(real((plan2.f).*exp.(λ γc))- real(p2),2) #/ max(maximum(abs.(real(γc))), maximum(abs.(real(γc_old))), 1)
		 if verbose
			 println("Iteration $count, err = ",  (err_βr+ err_γc))
		 end
		 if ((err_βr+ err_γc) < tol) || count > max_iter
			 break
		 end
		end 
	 
	end 
	
	return (f=βr,g=γc, count= count)

end
