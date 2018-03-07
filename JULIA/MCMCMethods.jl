"""
The following codes are meant to supplement the chapter 2.
"""
workspace()

using Gadfly
using Distributions: pdf, rand, Normal, Uniform
using DataFrames: DataFrame, nrow
using StatsBase: autocor
Gadfly.push_theme(:dark)

"""
LAPLACE APPROXIMATION
"""

function χ²(x::Array{Float64, 1}, ν::Int64)::AbstractArray{Float64, 1}
  x.^(ν - 1) .* exp(-(x.^2) ./ 2) ./ (2.^(ν/2 - 1) .* gamma(ν / 2))
end

k = [2, 10, 25, 30];
x = linspace(0, 6, 1000) |> Array;
for i in k
  y1 = χ²(x |> Array, i);
  y2 = pdf(Normal(sqrt(i - 1), sqrt(.5)), x);
  xyplot(x, y1, linestyle = "dashed", marker = "", xlab = "x", ylab = "Posterior + Approximator", save = false)
  xyplot(x, y2, linestyle = "solid", marker = "", add = true, file_name = joinpath(homedir(), "Dropbox/MS THESIS/CODES/JULIA/FIGURES", string("Chi_Post_Appr_", i, ".png")))
end

"""
MONTE CARLO SIMULATION
"""
draws = collect(100:1:10000);
area = Array{Float64, 1}();

for i in 1:(draws |> size)[1]
  samples = rand(Normal(0, 1), draws[i])
  push!(area, sum(((samples .> -1.96) & (samples .< 1.96)) ./ (samples |> size)[1]))
end

area_df = DataFrame(x = draws, y = area);
plot(area_df, x = :x, y = :y, Geom.line)

"""
METROPOLIS-HASTING ALGORITHM
"""
r = 10000;
burnIn = 500;
mu = [10; -10];
sigma = [[1.5^2  1.5*1.35*.5];
         [1.5*1.35*.5 1.35^2]];
x = Array{Float64, 1}(); push!(x, 30);

for i in 1:(r - 1)
  proposal = x[i] + rand(Uniform(-1, 1))
  accept = rand(Uniform()) < (pdf(Cauchy(), proposal) / pdf(Cauchy(), x[i]))

  if accept == true
    push!(x, proposal)
  else
    push!(x, x[i])
  end
end

x_df = DataFrame(it = collect(1:r), xs = x);
plot(x_df, x = :it, y = :xs, Geom.line)

x = repmat([NaN], r, 2);
x[1, :] = [0 0];
for i in 1:(r - 1)
  proposal = vec(x[i, :]) + vec(rand(Uniform(-5, 5), 2))
  accept = rand(Uniform()) < (pdf(MultivariateNormal(mu, sigma), proposal) / pdf(MultivariateNormal(mu, sigma), vec(x[i, :])))

  if accept == true
    x[i + 1, :] = proposal
  else
    x[i + 1, :] = x[i, :]
  end
end

x2_df = DataFrame(x);
plot(x2_df, x = :x1, y = :x2, Geom.path)



xyplot(x[(burnIn + 1):r, 1], x[(burnIn + 1):r, 2], marker = "o", linestyle = "")

"""
AUTOCORRELATION
"""
mh_acf1 = autocor(x[(burnIn + 1):r, 1]);
mh_acf2 = autocor(x[(burnIn + 1):r, 2]);

barplot(mh_acf1)
barplot(mh_acf2)

"""
GIBBS SAMPLING
"""
function cond_normal(x::Float64; μ1::Float64 = 10., μ2::Float64 = -10., σ1::Float64 = 1.5, σ2::Float64 = 1.35, ρ::Float64 = .5)
  rand(Normal(μ1 + (σ1/σ2) * ρ * (x - μ2), sqrt((1 - ρ^2) * σ1^2)))
end

x1 = Array{Float64, 1}(); push!(x1, 0.);
x2 = Array{Float64, 1}(); push!(x2, 0.);
for i in 1:r
  push!(x1, cond_normal(x2[i]))
  push!(x2, cond_normal(x1[i + 1], μ1 = -10., μ2 = 10., σ1 = 1.35, σ2 = 1.5))
end

xyplot(x1[(burnIn + 1):r], x2[(burnIn + 1):r], linestyle = "", marker = "o")

"""
AUTOCORRELATION
"""
gb_acf1 = autocor(x1[(burnIn + 1):r]);
gb_acf2 = autocor(x2[(burnIn + 1):r]);

barplot(gb_acf1)
barplot(gb_acf2)

"""
HAMILTONIAN MONTE CARLO
"""
immutable HMC
  U ::Function
  K ::Function
  dU::Function
  dK::Function
  d ::Int64
end

function hmc(Energies::HMC;
  leapfrog_params::Dict{Symbol, Real} = Dict([:ɛ => .3, :τ => 20]),
  set_seed::Int64 = 123,
  r::Int64 = 10000)

  """
  Energies - HMC type which contains functions for
             Potential and Kinetic Energies, including their
             corresponding gradients.
  leapfrog_params - parameters of the leap frog method
  r - number of sampling iteration
  """
  U, K, dU, dK, d = Energies.U, Energies.K, Energies.dU, Energies.dK, Energies.d
  ɛ, τ = leapfrog_params[:ɛ], leapfrog_params[:τ]
  H(x::AbstractArray{Float64, 1}, p::AbstractArray{Float64, 1}) = U(x) + K(p)

  if typeof(set_seed) == Int64
    srand(set_seed)
  end

  x = zeros(r, d);
  x[1, :] = zeros(d, 1);

  for i in 1:(r - 1)
    xNew = x[i, :]
    p = rand(Normal(), length(xNew))
    oldE = H(xNew, p)

    for j in 1:τ
      p = p - (ɛ / 2) * dU(xNew)
      xNew = xNew + ɛ * dK(p)
      p = p - (ɛ / 2) * dU(xNew)
    end

    newE = H(xNew, p)
    dE = newE - oldE

    if dE[1] < 0
      x[i + 1, :] = xNew
    elseif rand(Uniform()) < exp(-dE)[1]
      x[i + 1, :] = xNew
    else
      x[i + 1, :] = x[i, :]
    end
  end

  return x
end

"""
EXAMPLE: Sampling from Gaussian Distribution
"""
# Define the Functions
Potential(x::AbstractArray{Float64, 1}; μ::AbstractArray{Float64, 1} = zeros(2), Σ::AbstractArray{Float64, 2} = [[1  .9]; [.9 1]]) = (x - μ)' * (Σ)^(-1) * (x - μ);
dPotential(x::AbstractArray{Float64, 1}; μ::AbstractArray{Float64, 1} = zeros(2), Σ::AbstractArray{Float64, 2} = [[1  .9]; [.9 1]]) = (Σ)^(-1) * (x - μ);
Kinetic(p::AbstractArray{Float64, 1}) = (p' * p) / 2;
dKinetic(p::AbstractArray{Float64, 1}) = p;

HMC_object = HMC(Potential, Kinetic, dPotential, dKinetic, 2);

# Sample
@time weights = hmc(HMC_object, r = 50000);

# Plot the Samples
xyplot(weights[:, 1], weights[:, 2])
barplot(autocor(weights[:, 1]))
barplot(autocor(weights[:, 2]))

"""
STOCHASTIC GRADIENT HAMILTONIAN MONTE CARLO
"""
V = 1;

# Covariance Matrix
rho = 0.9;
covS = [[1 rho]; [rho 1]];
invS = inv( covS );

# Initial
x = [0; 0];

eta = 0.05;
alpha = 0.035;

# number of steps
L = 20;

funcU(x) = 0.5 * x'*invS*x;
gradUTrue(x) = invS * x;
gradUNoise(x) = invS * x  + randn(2,1);

function sghmc2d(gradU, eta, L, alpha, x, V)
  m = length(x)
  data = zeros(m, r -1)
  beta = V * eta * .5

  if beta > alpha
    error("too big eta")
  end

  sigma = sqrt(2 * eta * (alpha - beta))
  for n in 1:(r - 1)
    p = randn(m, 1) * sqrt(eta)
    momentum = 1 - alpha

    for t = 1:L
      p = p * momentum - gradU(x) * etaSGHMC + randn(2, 1) * sigma
      x = x + p
    end

    data[:, n] = x
  end

  return data
end
x
data
dsghmc = sghmc2d( gradUNoise, .005, 100, alpha, x, V );

xyplot(dsghmc[1, :], dsghmc[2, :])
barplot(autocor(dsghmc[1, 1000:end]))
barplot(autocor(dsghmc[2, 1000:end]))


"""
BAYESIAN LINEAR REGRESSION
"""
srand(1234);

# Set the parameters
w0 = -.3; w1 = -.5; stdev = 5.;

# Define data parameters
alpha = 1 / stdev; # for likelihood

# Generate Hypothetical Data
x = rand(Uniform(-1, 1), 20);
A = [ones(length(x)) x];
B = [w0; w1];
f = A * B
y = f + rand(Normal(0, alpha), 20)

# Define Hyperparameters
Imat = diagm(ones(2), 0)
b = 2 # for prior
b1 = (1 / b)^2 # Square this since in Julia, rnorm uses standard dev

mu = zeros(2) # for prior
s = b1 * Imat # for prior

# Prior
x1 = linspace(-1, 1, 200)
x2 = linspace(-1, 1, 200)

"""
BAYESIAN LINEAR REGRESSION USING METROPOLIS-HASTING

Previous section demonstrate the use of Bayesian linear regression obtain from derivation. In this section,
assuming we cannot integrate the normalizing factor of the posterior distribution and instead resort
into MCMC methods. The following codes generates the hypothetical data
"""
srand(1234);

# Set the parameters
w0 = -.8; w1 = -.1; stdev = 5.;

# Define data parameters
alpha = 1 / stdev; # for likelihood

# Generate Hypothetical Data
n = 200
x = rand(Uniform(-1, 1), n);
A = [ones(length(x)) x];
B = [w0; w1];
f = A * B
y = f + rand(Normal(0, alpha), n)

# Define Hyperparameters
Imat = diagm(ones(2), 0)
b = 2. # for prior
b1 = (1 / b)^2 # Square this since in Julia, rnorm uses standard dev

mu = zeros(2) # for prior
s = b1 * Imat # for prior

xyplot(x, y, xlab = "Predictor", ylab = "Response")

"""
The log likelihood function is given by the following codes:
"""
function loglike(theta::Array{Float64}; alpha::Float64 = alpha, x::Array{Float64} = x, y::Array{Float64} = y)
  yhat = theta[1] + theta[2]*x

  likhood = Float64[]
  for i in 1:length(yhat)
    push!(likhood, pdf(Normal(yhat[i], alpha), y[i]))
  end

  return likhood |> sum
end

"""
Define the log prior and lo posterior
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

"""
Define the proposal function and the metropolis-hasting algorithm
"""
sigmas = [1; 1];
function proposal(theta::Array{Float64})
  random = Float64[]

  for i in 1:length(theta)
    push!(random, rand(Normal(theta[i], sigmas[i])))
  end

  return random
end

function mh(theta_start, max_iter)
  chain = zeros((max_iter + 1, length(theta_start)))
  chain[1, :] = theta_start

  for i in 1:max_iter
    propose = proposal(chain[i, :])
    probab = exp(logpost(propose) - logpost(chain[i, :]))

    if rand(Uniform()) < probab
      chain[i + 1, :] = propose
    else
      chain[i + 1, :] = chain[i, :]
    end
  end

  return chain
end

"""
Apply the function
"""
@time mcmc = mh([0; 0], 50000)
w_est = mapslices(mean, mcmc, [1])

histogram(mcmc[:, 1])
histogram(mcmc[:, 2])

xyplot(mcmc[:, 1], mcmc[:, 2], linestyle = "-")

barplot(autocor(mcmc[:, 1]))
barplot(autocor(mcmc[:, 2]))
173.593244/23.297279

"""
BAYESIAN LINEAR REGRESSION USING HAMILTONIAN MONTE CARLO
"""
srand(123);
eps = .3
tau = 20

U(theta::Array{Float64}) = - logpost(theta)
dU(theta::Array{Float64}, alpha::Float64, x::Array{Float64}, y::Array{Float64}, b::Float64) = [-alpha * sum(y .- (theta[1] + theta[2] * x));
                             -alpha * sum((y .- (theta[1] + theta[2] * x)) .* x)] .+ b * theta

K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2
dK(p::Array{Float64}; Σ = eye(length(p))) = inv(Σ) * p
H(theta::Array{Float64}, p::Array{Float64}) = U(theta) + K(p)

chain = zeros((r, 2))
chain[1, :] = [-.78; -.75]

@time for n in 1:(r - 1)
  theta = chain[n, :]
  p = randn(length(theta))
  oldE = H(theta, p)

  for t_idx in 1:tau
    p = p - (eps / 2) * dU(theta, alpha, x, y, b)
    theta = theta + eps * dK(p)
    p = p - (eps / 2) * dU(theta, alpha, x, y, b)
  end

  newE = H(theta, p)
  dE = newE - oldE

  if dE[1] < 0
    chain[n + 1, :] = theta
  elseif rand(Uniform()) < exp(-dE[1])
    chain[n + 1, :] = theta
  else
    chain[n + 1, :] = chain[n, :]
  end
end
chain
mapslices(mean, chain, [1])

histogram(chain[:, 1])
histogram(chain[:, 2])

xyplot(chain[:, 1], chain[:, 2], linestyle = "-")

barplot(autocor(chain[:, 1]))
barplot(autocor(chain[:, 2]))


"""
BAYESIAN LINEAR REGRESSION USING STOCHASTIC GRADIENT HMC
"""
srand(123);
V = .1 * eye(2);

# Covariance Matrix
rho = 0.5;
covS = [[1 rho]; [rho 1]];
invS = inv(covS);

# Initial
theta = [0; 0.]

etaSGHMC = 0.05;
C = eye(2);

# number of steps
L = 20;

srand(123);

# Set the parameters
w0 = -.8; w1 = -.1; stdev = 5.;

# Define data parameters
alpha = 1 / stdev; # for likelihood

# Generate Hypothetical Data
n = 2000
x = rand(Uniform(-1, 1), n);
A = [ones(length(x)) x];
B = [w0; w1];
f = A * B
y = f + rand(Normal(0, alpha), n)

# Define Hyperparameters
Imat = diagm(ones(2), 0)
b = 2. # for prior
b1 = (1 / b)^2 # Square this since in Julia, rnorm uses standard dev

mu = zeros(2) # for prior
s = b1 * Imat # for prior

xyplot(x, y, xlab = "Predictor", ylab = "Response")

gradUNoise(x) = invS * x  + randn(2,1);
dU(theta::Array{Float64}; alpha::Float64 = alpha, x::Array{Float64} = x, y::Array{Float64} = y, b::Float64 = b) = [-alpha * sum(y - (theta[1] + theta[2] * x)); -alpha * sum((y - (theta[1] + theta[2] * x)) .* x)] + b * theta + randn(2,1)
eta = etaSGHMC
srand(123);
dU([0; 0.], alpha = alpha, x = x, y = y, b = b) |> print
function sghmc(gradU, eta, L, C, theta, V)
  m = length(theta)
  data = zeros(r -1, m)
  beta = V * eta * .5

  #if beta > C
#    error("too big eta")
  #end

  sigma = sqrt(2 * eta * (C - beta))
  n = 1
  for n in 1:(r - 1)
    p = randn(m, 1)
    for t = 1:L
      p = p - dU(theta, alpha = alpha, x = x, y = y, b = b) * eta - C * invS * p + sigma * randn(2, 1)
      theta = theta + dK(p) * eta
    end

    data[n, :] = theta
  end

  return data
end
theta
b = 2.
@time dsghmc = sghmc(dU, .05, 20, C, theta, V );
mapslices(mean, dsghmc, [1])
histogram(dsghmc[:, 1])
histogram(dsghmc[:, 2])
xyplot(dsghmc[:, 1], dsghmc[:, 2])

barplot(autocor(dsghmc[:, 1]))
barplot(autocor(dsghmc[:, 2]))





dU(theta, alpha = alpha, x = x, y = y, b = b)
