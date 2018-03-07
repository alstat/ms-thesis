using Gadfly
using Distributions
using NLreg
using RDatasets

pur = dataset("datasets", "Puromycin");
pur_treated = sub(pur, pur[:State] .== "treated");
pur_untreated = sub(pur, pur[:State] .== "untreated");
x = Array{Float64}(pur_treated[:Conc]);
y = Array{Float64}(pur_treated[:Rate]);

xy_data = DataFrame(X = x, Y = y);

plot(xy_data, x = :X, y = :Y, Coord.Cartesian(xmin = -0.05, xmax = 1.0))


"""
Define the Likelihood Function
"""
function loglike(theta::Array{Float64}; alpha::Float64 = alpha, x::Array{Float64} = x, y::Array{Float64} = y)
  yhat = (theta[1] * x) ./ (x + theta[2])

  likhood = Float64[]
  for i in 1:length(yhat)
    push!(likhood, log(pdf(Normal(yhat[i], 1 / alpha), y[i])))
  end

  return likhood |> sum
end

"""
Define the Prior Distribution
"""
function logprior(theta::Array{Float64}; w0_hypar::Array{Float64} = vm_hypar, w1_hypar::Array{Float64} = km_hypar)
  w0_prior = log(pdf(LogNormal(w0_hypar[1], 1 / w0_hypar[2]), theta[1]))
  w1_prior = log(pdf(LogNormal(w1_hypar[1], 1 / w1_hypar[2]), theta[2]))
   w_prior = [w0_prior w1_prior]

  return w_prior |> sum
end

"""
Define the Posterior Distribution
"""
function logpost(theta::Array{Float64})
  loglike(theta, alpha = alpha, x = x, y = y) + logprior(theta, w0_hypar = vm_hypar, w1_hypar = km_hypar)
end

"""
Define the Hyperparameters
"""
alpha =  1 / 3.;
beta1 =  1 / 1000.
beta2 =  1 / 10.
vm_hypar = [0; beta1];
km_hypar = [0; beta2];

"""
Apply Hamiltonian Monte Carlo
"""
U(theta::Array{Float64}) = - logpost(theta)
K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2
dK(p::AbstractArray{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;
function dU(theta::Array{Float64})
  [-beta1 * (1 / log(theta[1])) + alpha * sum(((y - (theta[1] * x) ./ (x + theta[2]))) .* (x ./ (theta[2] + x)));
   -beta2 * (1 / log(theta[2])) - alpha * sum(((y - (theta[1] * log(x)) ./ (x + theta[2]))) .* ((theta[1] * x) ./ ((theta[2] + x) .^ 2)))]
end
function dU_noise(theta::Array{Float64}; alpha::Float64 = 1/5., b::Float64 = 2.)
  [-beta1 * (theta[1]) + alpha * sum(((y - (theta[1] * log(x)) ./ (log(x) + theta[2]))) .* (log(x) ./ (theta[2] + log(x))));
   -beta2 * (theta[2]) - alpha * sum(((y - (theta[1] * log(x)) ./ (log(x) + theta[2]))) .* ((theta[1] * log(x)) ./ ((theta[2] + log(x)) .^ 2)))] + randn(2,1)
end

HMC_object2 = HMC(U, K, dU, dK, [200; .05], 2);

# Sample
@time weights = mcmc(HMC_object2, leapfrog_params = Dict([:ɛ => .009, :τ => 20]), r = 10000);
weights
SGHMC_object = SGHMC(dU_noise, dK, eye(2), eye(2), eye(2), [0; 0], 2.);

@time weights1 = mcmc(SGHMC_object, leapfrog_params = Dict([:ɛ => .009, :τ => 20]), r = 10000);
weights1
ws_df = DataFrame(w1 = weights1[:, 1], w2 = weights1[:, 2]);

plot(ws_df, x = :w2)
parameters = HMC_object2
leapfrog_params = Dict([:ɛ => 0.05, :τ => 20])
set_seed = 123
r = 1000

U, K, dU, dK, w, d = parameters.U, parameters.K, parameters.dU, parameters.dK, parameters.init_est, parameters.d
ɛ, τ = leapfrog_params[:ɛ], leapfrog_params[:τ]
H(x::AbstractArray{Float64}, p::AbstractArray{Float64}) = U(x) + K(p)
w
if typeof(set_seed) == Int64
  srand(set_seed)
end

chain = zeros(r, d);
chain[1, :] = w
i  = 1
for i in 1:(r - 1)
  w = chain[i, :]
  p = randn(length(w))
  oldE = H(w, p)

  for j in 1:τ
    p = p - (ɛ / 2) * dU(w)
    w = w + ɛ * dK(p)
    p = p - (ɛ / 2) * dU(w)
  end
#w
#p
  newE = H(w, p)
  dE = newE - oldE

  if dE[1] < 0
    chain[i + 1, :] = w
  elseif rand(Uniform()) < exp(-dE)[1]
    chain[i + 1, :] = w
  else
    chain[i + 1, :] = chain[i, :]
  end
end
chain
return chain
