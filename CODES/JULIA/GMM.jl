""" CODE FOR GAUSSIAN MIXTURE MODEL

This file attempts to generate do cluster analysis
using a Gaussian Mixture Model through the
Expectation-Maximization algorithm
"""
cd(joinpath(homedir(), "Dropbox/MS Thesis/Julia Files"))

workspace()

using Iterators
using Distributions
using PyPlot

function gmm(x::AbstractVector; k::Dict = Dict(:mix1 => 1/3, :mix2 => 1/3, :mix3 => 1/3),
  μ::Dict = Dict(:mix1 => [1; -1.5], :mix2 => [-1; 0], :mix3 => [2; 2]),
  Σ::Dict = Dict(:mix1 => [[.7 .2]; [.2 .5]], :mix2 => [[.6 .15]; [.15 .2]], :mix3 => [[.8 .15]; [.15 1.6]]))
  """ FUNCTION FOR DENSITY FUNCTION OF GAUSSIAN MIXTURE MODEL
  ------------------------------------------------------------------------------
  x - the input vector from the support of the GMM
  k - coefficient of the mixtures
  μ - mean of the mixtures
  Σ - variance-covariance matrices of the mixtures
  """

  temp = 0;
  for key in (k |> keys |> collect)
    temp += k[key] * pdf(MultivariateNormal(μ[key], Σ[key]), x)
  end

  return temp
end

function sim_gmm(n::Int; k::Dict = Dict(:mix1 => 1/3, :mix2 => 1/3, :mix3 => 1/3),
  μ::Dict = Dict(:mix1 => [1; -1.5], :mix2 => [-1; 0], :mix3 => [2; 2]),
  Σ::Dict = Dict(:mix1 => [[.7 .2]; [.2 .5]], :mix2 => [[.6 .15]; [.15 .2]], :mix3 => [[.8 .15]; [.15 1.6]]))
  """ FUNCTION FOR SIMULATING OF GAUSSIAN MIXTURE MODEL
  ------------------------------------------------------------------------------
  n - length of the simulated GMM
  k - coefficient of the mixtures
  μ - mean of the mixtures
  Σ - variance-covariance matrices of the mixtures
  """

  samples = zeros(2, n)
  for i in 1:n
    key = wsample(keys(k) |> collect, values(k) |> collect, 1)
    samples[:, i] = rand(MultivariateNormal(μ[key[1]], Σ[key[1]]), 1)
  end

  return samples
end

# C O N T O U R   P L O T   O F   G M M
n = 100;
x = y = linspace(-4, 5, n);
z = zeros((n, n));

xgrid = repmat(x, 1, n);
ygrid = repmat(y', n, 1);

for i in 1:n
  for j in 1:n
    z[i:i, j:j] = gmm([x[i]; y[j]])
  end
end

n = 1000;
d = sim_gmm(n);
graph(d'[:, 1], d'[:, 2], marker = "o");
xyplot(d'[:, 1], d'[:, 2], marker = "o");
d

col, bgcol, grdcol, lty, ticksize = ("#EFDECD", "#FAE7B5", "#696969", "dotted", 9)
graph(xgrid, ygrid, z, ptype = "contour")
graph(d'[:, 1], d'[:, 2], marker = "o", add = true)
# R A N D O M   S A M P L E S
# E S T I M A T I O N   U S I N G   E M
# GMM log-Likelihood
function gmm_loglik(y::AbstractArray; k::Dict = Dict(:mix1 => 1/3, :mix2 => 1/3, :mix3 => 1/3),
  μ::Dict = Dict(:mix1 => [1; -1.5], :mix2 => [-1; 0], :mix3 => [2; 2]),
  Σ::Dict = Dict(:mix1 => [[.7 .2]; [.2 .5]], :mix2 => [[.6 .15]; [.15 .2]], :mix3 => [[.8 .15]; [.15 1.6]]))
  """ GAUSSIAN MIXTURE MODELS LOG-LIKELIHOOD
  ------------------------------------------------------------------------------
  y - data array (d, n), where d is the dimension and n is the total obsrvations
  k - coefficient of the mixtures
  μ - mean of the mixtures
  Σ - variance-covariance matrices of the mixtures
  """

  log_lik = γ_den = 0
  for i in 1:(y |> size)[2]
    for mix in (μ |> keys |> collect)
      γ_den += (Θ[mix] * pdf(MultivariateNormal(μ[mix], Σ[mix]), y[:, i]))
    end
    log_lik += γ_den |> log
  end

  return log_lik
end

# E Step
function responsibility(y::AbstractArray, i::Int, j::Int; k::Dict = Dict(:mix1 => 1/3, :mix2 => 1/3, :mix3 => 1/3),
  μ::Dict = Dict(:mix1 => [1; -1.5], :mix2 => [-1; 0], :mix3 => [2; 2]),
  Σ::Dict = Dict(:mix1 => [[.7 .2]; [.2 .5]], :mix2 => [[.6 .15]; [.15 .2]], :mix3 => [[.8 .15]; [.15 1.6]]))
  """ RESPONSIBILITY FUNCTION
  ------------------------------------------------------------------------------
  y - data array (d, n), where d is the dimension and n is the total obsrvations
  i - index for the observations
  j - index for the mixtures (or keys of the dictionay object)
  k - coefficient of the mixtures
  μ - mean of the mixtures
  Σ - variance-covariance matrices of the mixtures
  """

  key = (μ |> keys |> collect)[j]
  γ_num = (Θ[key] * pdf(MultivariateNormal(μ[key], Σ[key]), y[:, i]))

  γ_den = 0
  for mix in (μ |> keys |> collect)
    γ_den += (Θ[mix] * pdf(MultivariateNormal(μ[mix], Σ[mix]), y[:, i]))
  end
  γ = γ_num / γ_den

  return γ
end

# INITIALIZATION
μ = Dict{Symbol, Array{Float64}}(:mix1 => [0; 0], :mix2 => [0; 0], :mix3 => [0; 0])
Σ = Dict{Symbol, Array{Float64}}(:mix1 => eye(2), :mix2 => eye(2), :mix3 => eye(2))
Θ = Dict{Symbol, Float64}(:mix1 => 1/3, :mix2 => 1/3, :mix3 => 1/3)
old_gmm_lik = gmm_loglik(d; k = Θ, μ = μ, Σ = Σ)
new_gmm_lik, q, maxq = Inf, 1, 1000
f = zeros(maxq)

# M Step
while ((new_gmm_lik - old_gmm_lik) |> abs) > .00000001
  old_gmm_lik = new_gmm_lik
  for j in 1:((μ |> keys |> collect) |> size)[1]
    γ_μnu, γ_Σnu, γ_de = 0, 0, 0
    for i in 1:(d |> size)[2]
      γ_μnu += responsibility(d, i, j; k = Θ, μ = μ, Σ = Σ) * d[:, i]
      γ_de += responsibility(d, i, j; k = Θ, μ = μ, Σ = Σ)
    end
    μ[(μ |> keys |> collect)[j]] = γ_μnu / γ_de

    for i in 1:(d |> size)[2]
      γ_Σnu += responsibility(d, i, j; k = Θ, μ = μ, Σ = Σ) * (d[:, i] - μ[(μ |> keys |> collect)[j]]) * (d[:, i] - μ[(μ |> keys |> collect)[j]])'
    end

    Σ[(Σ |> keys |> collect)[j]] = γ_Σnu / γ_de
    Θ[(Σ |> keys |> collect)[j]] = γ_de / (d |> size)[2]
  end

  new_gmm_lik = gmm_loglik(d; k = Θ, μ = μ, Σ = Σ)
  f[q] = (new_gmm_lik - old_gmm_lik) |> abs
  q += 1
  if q > maxq
    break
  end
end


f[1:69]
q
μ
Σ
Θ


# C O N T O U R   P L O T   O F   G M M
n = 100
x = y = linspace(-4, 5, n)
z = zeros((n, n))

xgrid = repmat(x, 1, n)
ygrid = repmat(y', n, 1)

for i in 1:n
  for j in 1:n
    z[i:i, j:j] = gmm([x[i]; y[j]])
  end
end

n = 100
x = y = linspace(-4, 5, n)
z1 = zeros((n, n))

for i in 1:n
  for j in 1:n
    z1[i:i, j:j] = gmm([x[i]; y[j]]; k = Θ, μ = μ, Σ = Σ)
  end
end
col, bgcol, grdcol, lty, ticksize = ("#EFDECD", "#FAE7B5", "#696969", "dotted", 9)
graph(xgrid, ygrid, z, ptype = "contour")
graph(xgrid, ygrid, z1, ptype = "contour")


clf(); cla(); close("all")
f, ax = subplots(figsize = (5, 5))
ax[:plot](d[1, :]', d[2, :]', marker = "o", linestyle = "None")
ax[:contour](xgrid, ygrid, z, linewidth = 2.0)
while ((new_gmm_lik - old_gmm_lik) |> abs) > 1
  old_gmm_lik = new_gmm_lik
  for j in 1:((μ |> keys |> collect) |> size)[1]
    γ_μnu, γ_Σnu, γ_de = 0, 0, 0
    for i in 1:(d |> size)[2]
      γ_μnu += responsibility(d, i, j; k = Θ, μ = μ, Σ = Σ) * d[:, i]
      γ_de += responsibility(d, i, j; k = Θ, μ = μ, Σ = Σ)
    end
    μ[(μ |> keys |> collect)[j]] = γ_μnu / γ_de

    for i in 1:(d |> size)[2]
      γ_Σnu += responsibility(d, i, j; k = Θ, μ = μ, Σ = Σ) * (d[:, i] - μ[(μ |> keys |> collect)[j]]) * (d[:, i] - μ[(μ |> keys |> collect)[j]])'
    end

    Σ[(Σ |> keys |> collect)[j]] = γ_Σnu / γ_de
    Θ[(Σ |> keys |> collect)[j]] = γ_de / (d |> size)[2]
  end

  new_gmm_lik = gmm_loglik(d; k = Θ, μ = μ, Σ = Σ)
  f[q] = (new_gmm_lik - old_gmm_lik) |> abs
  q += 1
  if q > maxq
    break
  end
end


f
f[100:200]
(new_gmm_lik - old_gmm_lik) |> abs
old_gmm_lik = new_gmm_lik

μ
Σ
Θ

γ_Σnu


0 + μ[:mix1]
μ = [0; 0]
d
for i in 1:(d |> size)[2]

  μ += responsibility(d, i, 1; k = Θ, μ = μ, Σ = Σ)
end

γ = responsibility(d, 2, 3; k = Θ, μ = μ, Σ = Σ)
γ = gmm_loglik(d)
while δ > ν



responsibility(d, 4, 3; k = Θ, μ = μ, Σ = Σ)
gmm_loglik(d, k = Θ, μ = μ, Σ = Σ)


δ = Inf; ν = .001
while dif > ν







grid[:][:, 1]
for i, j in grid
  print(grid[])
end
ndg


x = [1,2,3]
y = ["a","b"]
z = [10,12]
d = collect(product(x,z))


gmm(x)
y


μ = [0; 0]
Σ = eye(2)
y = rand(MultivariateNormal(μ, Σ), 3)
x = [.5; .5]


zeros()
help
isa(MultivariateNormal(), Distributions.MultivariateDistribution)
isa(Normal(), Distribution)
isa(MultivariateNormal(μ, Σ), MultivariateDistribution)
function sample_mgmm(n::Int, k::AbstractVector; μ::AbstractVector = (k |> size)[1] |> zeros, Σ::AbstractArray = (k |> size)[1] |> eye)
  k_size = (k |> size)[1]
  mixture = [string("mix", i) for i in 1:k_size]

  for i in 1:n
    d = wsample(mixture, k, 1)[1]


  end
end


  MixtureModel(map(μ -> Normal(μ, eye(2)), [[1; 1]; [2; 2]]))
function GaussianMultivariateMixture()

end
