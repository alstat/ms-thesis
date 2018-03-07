using Plots
using Distributions: pdf, rand, Normal, Uniform
using DataFrames: DataFrame, nrow
using StatsBase: autocor
using GLM
using NLreg
using RDatasets
using DataFrames
using StochMCMC

"""
I.  SAMPLING FROM GAUSSIAN DISTRIBUTION
"""
# Define the Functions
potential(x::AbstractArray{Float64}; μ::AbstractArray{Float64} = [10.; 10], Σ::AbstractArray{Float64} = eye(2)) = (x - μ)' * (Σ)^(-1) * (x - μ);
dpotential(x::AbstractArray{Float64}; μ::AbstractArray{Float64} = [10.; 10], Σ::AbstractArray{Float64} = eye(2)) = (Σ)^(-1) * (x - μ);
kinetic(p::AbstractArray{Float64}; Σ::Array{Float64} = eye(length(p))) = (p' * inv(Σ) * p) / 2;
dkinetic(p::AbstractArray{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;

"""
  a. HAMILTONIAN MONTE CARLO
"""
HMC_object = HMC(potential, kinetic, dpotential, dkinetic, [0; 0], 2);
@time weights = mcmc(HMC_object);

# Plot the Samples
mapslices(mean, weights, [1]) |> print

weights_df = DataFrame(weights);
autcorw_df = DataFrame(x1 = autocor(weights[:, 1]), x2 = autocor(weights[:, 2]));
plot(weights_df, x = :x1, Geom.histogram);
plot(weights_df, x = :x2, Geom.histogram);
plot(weights_df, x = :x1, y = :x2, Geom.point);
plot(autcorw_df, x = collect(0:(nrow(autcorw_df) - 1)), y = :x1, Geom.bar, Coord.cartesian(xmin = -2, xmax = 32, ymin = -.1, ymax = 1.1))

"""
  a. STOCHASTIC GRADIENT HAMILTONIAN MONTE CARLO
"""
SGHMC_object = SGHMC(dpotential, dkinetic, eye(2), eye(2), eye(2), [0; 0], 2);
@time weights1 = sghmc(SGHMC_object);

# Plot the Samples
mapslices(mean, weights1, [1])

histogram(weights1[:, 1])
histogram(weights1[:, 2])

xyplot(weights1[:, 1], weights1[:, 2])

barplot(autocor(weights1[:, 1]))
barplot(autocor(weights1[:, 2]))

"""
II. BAYESIAN LINEAR REGRESSION
"""
srand(123);
w0 = .2; w1 = -.9; stdev = 5.;

# Define data parameters
alpha = 1 / stdev; # for likelihood

# Generate Hypothetical Data
n = 200
x = rand(Uniform(-1, 1), n);
A = [ones(length(x)) x];
B = [w0; w1];
f = A * B
y = f + rand(Normal(0, alpha), n)

#pur = dataset("datasets", "Puromycin");
#pur_treated = sub(pur, pur[:State] .== "treated");
#x = Array(pur_treated[:Conc]);
#y = Array{Float64}(pur_treated[:Rate]);

#y = 3 * exp(x / .5) + rand(Normal(0, 1), length(x))

#scatter(x, y)
# Define Hyperparameters
Imat = diagm(ones(2), 0)
b = 2. # for prior
b1 = (1 / b)^2 # Square this since in Julia, rnorm uses standard dev

mu = zeros(20) # for prior
s = b1 * Imat # for prior

"""
The log likelihood function is given by the following codes:
"""
function loglike(theta::Array{Float64}; alpha::Float64 = alpha, x::Array{Float64} = x, y::Array{Float64} = y)
  yhat = theta[1] * exp(x / theta[2])

  likhood = Float64[]
  for i in 1:length(yhat)
    push!(likhood, log(pdf(Normal(yhat[i], alpha), y[i])))
  end

  return likhood |> sum
end

"""
Define the log prior and log posterior
"""
function logprior(theta::Array{Float64}; mu::Array{Float64} = mu, s::Array{Float64} = s)
  w0_prior = log(pdf(Normal(mu[1, 1], s[1, 1]), theta[1]))
  w1_prior = log(pdf(Normal(mu[2, 1], s[2, 2]), theta[2]))
   w_prior = [w0_prior w1_prior]

  return w_prior |> sum
end

function logpost(theta::Array{Float64})
  loglike(theta, alpha = alpha, x = x, y = y) + logprior(theta, mu = mu, s = s)
end

U(theta::Array{Float64}) = - logpost(theta)
K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2
function dPotential(theta::Array{Float64}; alpha::Float64 = 1/5., b::Float64 = 2.)
  [-alpha * sum(y - (theta[1] + theta[2] * x));
   -alpha * sum((y - (theta[1] + theta[2] * x)) .* x)] + b * theta
end
function dPotential_noise(theta::Array{Float64}; alpha::Float64 = 1/5., b::Float64 = 2.)
  [-alpha * sum(y - (theta[1] + theta[2] * x));
   -alpha * sum((y - (theta[1] + theta[2] * x)) .* x)] + b * theta + randn(2,1)
end

dKinetic(p::AbstractArray{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;

function dPot(theta::Array{Float64}; alpha::Float64 = 1/5., b::Float64 = 2.)
  [-beta1 * (theta[1] - 10) - sum(alpha * (y - (x ./ (theta[2] + x))));
   -beta2 * (theta[2] - 10) - sum(alpha * (y + ((theta[1] * x) ./ ((theta[2] + x) .^ 2))))]
end

function dPot_noise(theta::Array{Float64}; alpha::Float64 = 1/5., b::Float64 = 2.)
  [-beta1 * (theta[1] - 10) - sum(alpha * (y - (x ./ (theta[2] + x))));
   -beta2 * (theta[2] - 10) - sum(alpha * (y + ((theta[1] * x) ./ ((theta[2] + x) .^ 2))))] + randn(2,1)
end
theta = [0.; 0]

-beta2 * (theta[2] - .2 * maximum(x)) - sum(alpha * (y + ((theta[1] * x) ./ ((theta[2] + x) ^ 2))))
function dPot2(theta::Array{Float64})
  [1]
end
"""
  a. Metropolis-Hasting
"""
mh_object = MH(logpost; init_est = mu);
@time chain1 = mcmc(mh_object, r = 10000);
est = mapslices(mean, chain1, [1])
z = (est[1] * x) ./ (est[2] + x)

scatter(x, y)
plot!(x, z)
"""
  a. HAMILTONIAN MONTE CARLO
"""
HMC_object2 = HMC(U, K, dPotential, dKinetic, zeros(2, 1), 2);

# Sample
@time weights = mcmc(HMC_object2, leapfrog_params = Dict([:ɛ => .009, :τ => 20]), r = 10000);
weights

# Plot the Samples
mapslices(mean, weights, [1]) |> print
weights_df = DataFrame(weights);

plot(weights_df, x = :x1, Geom.histogram)
plot(weights_df, x = :x2, Geom.histogram)
plot(weights_df, x = :x1, y = :x2, Geom.point)

gewekediag(weights[:, 1])
gewekediag(collect(1:100))
xyplot(weights[:, 1], weights[:, 2])

barplot(autocor(weights[:, 1]))
barplot(autocor(weights[:, 2]))

"""
  b. STOCHASTIC GRADIENT HAMILTONIAN MONTE CARLO
"""
SGHMC_object = SGHMC(dPot_noise, dKinetic, eye(2), eye(2), eye(2), [0; 0], 2.);

@time weights1 = mcmc(SGHMC_object, leapfrog_params = Dict([:ɛ => .0009, :τ => 20]), r = 10000);
mapslices(mean, weights1[(!isnan(weights1[:, 1]) |> find), :], [1]) |> print

histogram(weights1[:, 1])
histogram(weights1[:, 2], nbins = 20)

xyplot(weights1[:, 1], weights1[:, 2])

barplot(autocor(weights1[:, 1]))
barplot(autocor(weights1[:, 2]))

"""
II. BAYESIAN NONLINEAR REGRESSION
"""

pur = dataset("datasets", "Puromycin");
pur_treated = sub(pur, pur[:State] .== "treated");

# Estimate the parameters using Nonlinear Least Square

# Estimate using Linear Least Square
new_dat = DataFrame(x = 1 / pur_treated[:Conc], y = 1 / pur_treated[:Rate]);
scatter(new_dat[:x], new_dat[:y])

ols = glm(y ~ x, new_dat, Normal(), IdentityLink());
ols1 = glm(Rate ~ Conc, pur_treated, Normal(), InverseLink());
v_m = 1 / coef(ols)[1]

km = v_m * coef(ols)[2]
z = Array((v_m * pur_treated[:Conc]) ./ (km + pur_treated[:Conc]));
scatter(pur_treated[:Conc], pur_treated[:Rate])
plot!(pur_treated[:Conc], z)
z
ols     b  

pm1 = fit(MicMen(Rate ~ Conc, pur_treated), true);


x = Array(pur_treated[:Conc]);
y = Array{Float64}(pur_treated[:Rate]);

x = collect(minimum(pur_treated[:Conc]):.01:maximum(pur_treated[:Conc]));
#y = (200 * x) ./ (.05 + x) + randn(length(x))
y = 200 * (x .^(2)) + rand(Normal(0, 10), length(x))
scatter(x, y)

gamma1 = 0.1 * maximum(x);
gamma2 = maximum(y);
beta1 = beta2 = 1 / 4;

mu = [gamma1; gamma2];
s = beta1 * eye(2);

alpha = 1 / 5.;
"""
The log likelihood function is given by the following codes:
"""

function loglike(theta::Array{Float64}; alpha::Float64 = alpha, x::Array{Float64} = x, y::Array{Float64} = y)
  #yhat = (theta[1] * x) ./ (theta[2] + x)
  #yhat = theta[1] * x .^ (theta[2])
  yhat = theta[1] + theta[2]*x

  likhood = Float64[]
  for i in 1:length(yhat)
    push!(likhood, log(pdf(Normal(yhat[i], alpha), y[i])))
  end

  return likhood |> sum
end

"""
Define the log prior and log posterior
"""
function logprior(theta::Array{Float64}; mu::Array{Float64} = mu, s::Array{Float64} = s)
  w0_prior = log(pdf(Normal(mu[1, 1], s[1, 1]), theta[1]))
  w1_prior = log(pdf(Normal(mu[2, 1], s[2, 2]), theta[2]))
   w_prior = [w0_prior w1_prior]

  return w_prior |> sum
end

function logpost(theta::Array{Float64})
  loglike(theta, alpha = alpha, x = x, y = y) + logprior(theta, mu = mu, s = s)
end

mh_object = MH(logpost; init_est = mu);
@time chain1 = mcmc(mh_object, r = 10000);

est = mapslices(mean, chain1, [1])
z = (est[1] * x) ./ (est[2] + x)
z = est[1] * x .^ est[2]

xyz_df = DataFrame(X = x, Y = y, Z = z);
scatter(x, y)
plot!(x, z)

plot(layer(xyz_df, x = :X, y = :Y, Geom.point),
     layer(xyz_df, x = :X, y = :Z, Geom.line))


.9
