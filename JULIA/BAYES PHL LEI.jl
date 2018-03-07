using DataFrames
using Distributions
using Gadfly
srand(123)

in_dir = joinpath(homedir(), "Dropbox/MS THESIS/JULIA/INPUT");
lei_data = readtable(joinpath(in_dir, "Cycles LEI Data.csv"));
indicators =  lei_data[:, collect(2:8)];
x = Array(indicators[:, 2:7]);
y = Array(indicators[:, 1]);

series = DataFrame(Y = y);
plot(series, y = :Y, Geom.line())

"""
The log likelihood function is given by the following codes:
"""
function loglike(theta::Array{Float64})
  yhat = 0
  for i in 1:(size(x)[2])
    yhat += theta[i] * x[:, i]
  end

  likhood = Float64[]
  for i in 1:length(yhat)
    push!(likhood, log(pdf(Normal(yhat[i], alpha), y[i])))
  end

  return likhood |> sum
end

"""
Define the Prior Distribution
"""
function logprior(theta::Array{Float64})
  w_prior = [log(pdf(Normal(mu[i], s[i]), theta[i])) for i in 1:(size(x)[2])]

  return w_prior |> sum
end

"""
Define the Posterior Distribution
"""
function logpost(theta::Array{Float64})
  loglike(theta) + logprior(theta)
end

"""
Do Bayesian Estimation using Metropolis-Hasting
"""
alpha = 1 / 5.;
mu = zeros(size(x)[2]);
s = ones(size(x)[2]);

mh_object = MH(logpost; init_est = ones(6), d = 6);
@time chain1 = mcmc(mh_object, r = 10000);
chain1
"""
Plot it
"""
est1 = mapslices(mean, chain1[1000:end, :], [1]);
est1
yhat = 0
for i in 1:(size(x)[2])
  yhat += est1[i] * x[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)
series[:yhat_std] = yhat_std;

for i in 1:size(chain1[1000:end, :])[1]

  for j in 1:(size(x)[2])
    yhat += chain1[i, j] * x[:, j]
  end

  series[Symbol("yhat_std_", string(i))] = (yhat - mean(yhat)) / std(yhat)

end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series)), outer = size(chain1[1000:end, :])[1]),
  var = Array(stack(series[:, 3:end])[1]),
  val = Array(stack(series[:, 3:end])[2])
  );

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"gray")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"orange")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )

"""
Do Bayesian Estimation using Hamiltonian Monte Carlo
"""
U(theta::Array{Float64}) = - logpost(theta)
K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2
function dU(theta::Array{Float64})
  yhat = 0
  for i in 1:size(x)[2]
    yhat += theta[i] * x[:, i]
  end
  [-alpha * sum((y - yhat) .* x[:, i]) + s[i] * theta[i] for i in 1:size(x)[2]]
end
dK(p::Array{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;

HMC_object2 = HMC(U, K, dU, dK, zeros(6, 1), 6);

# Sample
@time chain2 = mcmc(HMC_object2, leapfrog_params = Dict([:ɛ => .009, :τ => 20]), r = 10000);

est2 = mapslices(mean, chain2[1000:end, :], [1])
yhat = 0
for i in 1:(size(x)[2])
  yhat += est2[i] * x[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)
series[:yhat_std] = yhat_std;

for i in 1:size(chain2[1000:end, :])[1]

  for j in 1:(size(x)[2])
    yhat += chain2[i, j] * x[:, j]
  end

  series[Symbol("yhat_std_", string(i))] = (yhat - mean(yhat)) / std(yhat)

end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series)), outer = size(chain2[1000:end, :])[1]),
  var = Array(stack(series[:, 3:end])[1]),
  val = Array(stack(series[:, 3:end])[2])
  );

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"gray")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"orange")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )

"""
Do Bayesian Estimation using Stochastic Gradient Hamiltonian Monte Carlo
"""
U(theta::Array{Float64}) = - logpost(theta)
K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2
function dU_noise(theta::Array{Float64})
  yhat = 0
  for i in 1:size(x)[2]
    yhat += theta[i] * x[:, i]
  end
  [-alpha * sum((y - yhat) .* x[:, i] + s[i] * theta[i]) for i in 1:size(x)[2]] + randn(size(x)[2])
end
dK(p::Array{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;

SGHMC_object = SGHMC(dU_noise, dK, eye(6), eye(6), eye(6), zeros(6), 6.);

@time chain3 = mcmc(SGHMC_object, leapfrog_params = Dict([:ɛ => .0009, :τ => 20]), r = 10000);
est3 = mapslices(mean, chain3[1000:end, :], [1])

yhat = 0
for i in 1:(size(x)[2])
  yhat += est3[i] * x[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)
series[:yhat_std] = yhat_std;

for i in 1:size(chain3[1000:end, :])[1]

  for j in 1:(size(x)[2])
    yhat += chain3[i, j] * x[:, j]
  end

  series[Symbol("yhat_std_", string(i))] = (yhat - mean(yhat)) / std(yhat)

end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series)), outer = size(chain3[1000:end, :])[1]),
  var = Array(stack(series[:, 3:end])[1]),
  val = Array(stack(series[:, 3:end])[2])
  );

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"gray")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"orange")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )
