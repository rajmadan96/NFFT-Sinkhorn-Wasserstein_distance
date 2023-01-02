using LinearAlgebra
using JuMP, Tulip			# Coin-or linear programming
using LogExpFunctions
include("ontoSimplex.jl")


function dFunction(s1::T, s2::T; p=2) where T
	norm(s2- s1, p)
end

function distFunction(states1::Array{Float64}, states2::Array{Float64})::Array{Float64,2}
	n1= size(states1, 1)		# works in various dimensions
	n2= size(states2, 1)
	dMatrix= Array{Float64}(undef, (n1,n2))
	for i= 1:n1, j= 1:n2		# Euclidean norm
		dMatrix[i,j]= norm(states2[j,:]- states1[i,:])
	end
	return dMatrix
end

function Wasserstein(p1::Vector{Float64}, p2::Vector{Float64}, distMatrix::Array{Float64,2}, rWasserstein::Float64=1.)
	ontoSimplex!(p1); ontoSimplex!(p2)
    n1= length(p1);   n2= length(p2)

    A= kron(ones(n2)', Matrix{Float64}(I, n1, n1))
    B= kron(Matrix{Float64}(I, n2, n2), ones(n1)')

	model = Model(Tulip.Optimizer)
	@variable(model, x[i=1:n1*n2] >= 0)
	@objective(model, Min, vec(distMatrix.^rWasserstein)' * x)
	@constraint(model, [A;B] * x .== [p1;p2])
	optimize!(model)
    return (distance= (objective_value(model))^(1/ rWasserstein), π= reshape(value.(x), (n1, n2)))
end

function Sinkhorn(p1::Vector{Float64}, p2::Vector{Float64}, distMatrix::Array{Float64,2}; rWasserstein::Float64= 1., λ::Float64= 1.,tol=1e-5,max_iter=1000,verbose=false)
	ontoSimplex!(p1); βr= Vector{Float64}(undef, length(p1))
	ontoSimplex!(p2); γc= ones(size(p2))		# guess a starting value
	count= 0; 
	distMatrix.^= rWasserstein; K= exp.(-λ * distMatrix)
	while true 	# Sinkhorn iteration
		γc= γc./ (p2'*γc)		# rescale
		βr= p1./ (K* γc)		# vector operation
		tmp= K'* βr
		γc= p2./ tmp			# vector operation
		count += 1;
		if count % 10 == 0
			err_βr=
				 norm(real((K* γc).* βr)- real(p1),2) #/ max(maximum(abs.(real(βr))), maximum(abs.(real(βr_old))), 1)
			 err_γc=
				 norm(real(tmp.*γc)- real(p2),2) #/ max(maximum(abs.(real(γc))), maximum(abs.(real(γc_old))), 1)
			 if verbose
				 println("Iteration $count, err = ",  (err_βr+ err_γc))
				 end
			 if ((err_βr+ err_γc) < tol) || count > max_iter
				 break
			 end
		 end
	end
	tmp=1+p1'*replace!(log.(βr),-Inf=>0) + p2'*replace!(log.(γc),-Inf=>0)- tmp'* γc
	return (distSinkhorn= tmp/ λ,distSinkhornUB= (tmp-p1'*replace!(log.(p1),-Inf=>0)-p2'*replace!(log.(p2),-Inf=>0))/ λ,
			count= count)
end

	
function logSinkhorn_stablized(p1::Vector{Float64}, p2::Vector{Float64}, distMatrix::Array{Float64,2}; rWasserstein::Float64= 1., λ::Float64= 1.,tol=1e-5,max_iter=1000,verbose=false)
	ontoSimplex!(p1); βr=zeros(size(p1))
	ontoSimplex!(p2); γc= zeros(size(p2))		# guess a starting value
	count= 0; 
	distMatrix.^= rWasserstein; K= exp.(-λ * distMatrix)
	while true	# Sinkhorn iteration
		βr= (log.(p1.+ 1e-20).- vec(logsumexp((λ *(- distMatrix.+γc'));dims=2)))./λ		# vector operation

		γc= (log.(p2.+ 1e-20).- vec((logsumexp((λ *(- distMatrix.+βr ));dims=1))))./λ		# vector operation
		count += 1;
		if count % 10 == 0
			err_βr=
				norm(real((K* exp.(λ*γc)).* exp.(λ*βr))- real(p1),2) 
			err_γc=
				norm(real((K'* exp.(λ*βr)).*exp.(λ*γc))- real(p2),2) 
			if verbose
				println("Iteration $count, err = ",  (err_βr+ err_γc))
			end
			if ((err_βr+ err_γc) < tol) || count > max_iter
				break
			end
		end
	end
	f=((βr));g=((γc));
	return (f=βr,g=γc,count=count)
end



