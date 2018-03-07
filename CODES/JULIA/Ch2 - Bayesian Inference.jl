# ----------------------------------------------------------------------------
# Bayesian Inference for Artificial Neural Networks and Hidden Markov Models
#
# The following
# Author: Al-Ahmadgaid Asaad
# alasaadstat@gmail.com
#
#
# March 23, 2016
# ----------------------------------------------------------------------------
using StatsBase
using Distributions

# Monte-Carlo Integration
# -----------------------------
# Suppose we want to integrate the area of the normal curve
# between -1.96 and 1.96. And assuming it is difficult to
# carry out the integration. We can use Monte-Carlo Integration
# as an approximator which uses sampling. And as the sample goes
# large it can better approximate the distribution.
draws = collect(1000:100:100000)
area = zeros(length(draws))

for i = 1:length(draws)
  samples = rand(Normal(), draws[i])
  area[i] = sum((samples .> -1.96) & (samples .< 1.96)) / length(samples)
end

# Metropolis-Hasting Algorithm
# ---------------------------------
# Target Distribution: Univariate Cauchy
r = 100000
chain = zeros(r)
chain[1] = 30

for i = 1:r - 1
  proposal = chain[i] + rand(Uniform(-1, 1))
  accept = rand(Uniform()) < pdf(Cauchy(), proposal) / pdf(Cauchy(), chain[i])

  if accept == true
    chain[i + 1] = proposal
  else
    chain[i + 1] = chain[i]
  end

end

# Check the mixing of the samples

# Define Bivariate Cauchy
function BvCauchy(x, μ = [0; 0], γ = 1)
  (1 / (2pi)) * (γ / ((x[1] - μ[1])^2 + (x[2] - μ[2])^2 + γ^2)^1.5)
end

chain = zeros((2, r))
chain[:, 1] = [-100; 100]

for i = 1:r - 1
  proposal = chain[:, i] + rand(Uniform(-5, 5), 2)
  accept = rand(Uniform(), 1)[1] < (BvCauchy(proposal) / BvCauchy(chain[:, i]))
  if accept == true
    chain[:, i + 1] = proposal
  else
    chain[:, i + 1] = chain[i]
  end
end

# Hamiltonian Monte Carlo
# ----------------------------------

δ = .3
n = 10000
L = 20

# Define Potential Energy Function
U(x, Σ) = x' * inv(Σ) * x

# Gradient of Potential Energy Function
dU(x, Σ) = inv(Σ) * x

# Define Kinetic Energy
K(p) = (p' * p) / 2

# Initial State
x = zeros(2, n)
x0 = [0, 6]
x[:, 1] = x0

# Sigma for dU
Σ = [1 .8; .8 1]

t = 1
while t < n
  t = t + 1

  # Simulate Random Momentum
  p0 = rand(Normal(), 2)

  # Simulate Hamiltonian Dynamics
  # --------------------------------
  pStar = p0 - (δ / 2) * dU(x[:, t - 1], Σ)

  # Simulate
  xStar = x[:, t - 1] + δ * pStar;

  # Full Steps
  for jL = 1:L - 1

    # Momentum
    pStar = pStar - δ * dU(pStar, Σ)

    # Position
    xStar = xStar + δ * pStar

  end

  # Last Half Step
  pStar = pStar - (δ / 2) * dU(pStar, Σ)

  U0 = U(x[:, t - 1], Σ)
  UStar = U(xStar, Σ)

  K0 = K(p0)
  KStar = K(pStar)

  α = float(min(1, exp((U0 + K0) - (UStar + KStar))))

  u = rand(Uniform())

  if u < α[1]
    x[:, t] = xStar
  else
    x[:, t] = x[:, t - 1]
  end

end

# Plot the data points
