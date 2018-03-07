using DataFrames
using Distributions
using Gadfly
Gadfly.push_theme(:dark)

srand(123);

include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/MH.jl"));
include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/HMC.jl"));
include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/SG HMC.jl"));

in_dir = joinpath(homedir(), "Dropbox/MS THESIS/JULIA/INPUT");
ou_Rdir = joinpath(homedir(), "Dropbox/MS THESIS/R/INPUT");
lei_data = readtable(joinpath(in_dir, "SA LEI.csv"));
lei_growth = zeros(nrow(lei_data) - 1, ncol(lei_data) - 1);

for i in 2:ncol(lei_data)
  growth = diff(lei_data[i]) ./ lei_data[i][1:(end - 1)]
  lei_growth[:, i - 1] = (growth - mean(growth)) ./ std(growth)
end

lei_df = DataFrame(lei_growth);
plot(lei_df, y = :x1, Geom.line)
plot(lei_data, y = :GDP, Geom.line)

indicators =  lei_df;



function rmse(true_value::Array{Float64}, estimated_value::Array{Float64})
  sqrt((1 / length(true_value)) * sum((true_value - estimated_value) .^ 2))
end

"""
DATA PARTIONING
"""

# Training Data
n_percent = .7
n_train = Int(nrow(indicators) * n_percent |> floor)
x = Array(indicators[1:n_train, 2:7]);
y = Array(indicators[1:n_train, 1]);

series = DataFrame(Y = y);
plot(series, y = :Y, Geom.line, Guide.ylabel("Reference Series"))

# Testing Data
x_test = Array(indicators[(n_train + 1):end, 2:7]);
y_test = Array(indicators[(n_train + 1):end, 1]);

"""
ARDL(1, 1)
"""
y_1 = [mean(y[1:(end - 1)]); y[1:(end - 1)]];
x_1 = [mapslices(mean, x[1:(end - 1), :], [1]); x[1:(end - 1), :]];
n_params = 1 + length(size(y_1)) + size(x)[2] + size(x_1)[2]

y_1test = [mean(y_test[1:(end - 1)]); y_test[1:(end - 1)]];
x_1test = [mapslices(mean, x_test[1:(end - 1), :], [1]); x_test[1:(end - 1), :]];

"""
The log likelihood function is given by the following codes:
"""
function loglike(theta::Array{Float64})
  yhat = theta[1] + theta[2] * y_1
  for i in 1:(size(x)[2])
    yhat += theta[i + 2] * x[:, i] + theta[i + 2 + size(x)[2]] * x_1[:, i]
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
  w_prior = [log(pdf(Normal(mu[i], s[i]), theta[i])) for i in 1:(n_params)]

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
mu = zeros(n_params);
s = ones(n_params);
n_chain = 4
R = 100000
chain1 = Dict{Symbol, Array{Float64}}()

srand(123);
for i in 1:n_chain
  init_val = rand(Uniform(.001 + (i - 1) / 4, i / 4), n_params)
  mh_object = MH(logpost; init_est = init_val, d = n_params);
  chain1[Symbol("Realization_" * string(i))] = mcmc(mh_object, r = R);
end

for i in 1:n_chain
  writetable(joinpath(ou_Rdir, "MH Chain " * string(i) * ".csv"), DataFrame(chain1[Symbol("Realization_" * string(i))]))
end


"""
Plot it
"""
series = DataFrame(Y = y);
burnIn = 10;
stepsize = 10;

# Training Data
chain1_est = chain1_std = zeros(n_chain, n_params);
for i in 1:n_chain
  chain1_est[i, :] = mapslices(mean, chain1[Symbol("Realization_" * string(i))][(burnIn + 1):stepsize:end, :], [1]);
  chain1_std[i, :] = mapslices(std, chain1[Symbol("Realization_" * string(i))][(burnIn + 1):stepsize:end, :], [1])
end

chain1

est1 = mapslices(mean, chain1_est, [1]);
std1 = mapslices(mean, chain1_std, [1]);

chain1_ave = zeros(R, n_params);
for i in 1:R
  chain1_vec = [chain1[collect(keys(chain1))[1]][i, :] chain1[collect(keys(chain1))[2]][i, :] chain1[collect(keys(chain1))[3]][i, :] chain1[collect(keys(chain1))[4]][i, :]]'
  chain1_ave[i, :] = mapslices(mean, chain1_vec, [1])
end

writetable(joinpath(ou_Rdir, "MH Chain Ave.csv"), DataFrame(chain1_ave))

yhat = est1[1] + est1[2] * y_1
for i in 1:(size(x)[2])
  yhat += est1[i + 2] * x[:, i] + est1[i + 2 + size(x)[2]] * x_1[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)

series[:yhat_std] = yhat_std;

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red"))
  )

for i in (burnIn + 1):stepsize:size(chain1_ave)[1]
  yhat = chain1_ave[i, 1] + chain1_ave[i, 2] * y_1
  for j in 1:(size(x)[2])
    yhat += chain1_ave[i, j + 2] * x[:, j] + chain1_ave[i, j + 2 + size(x)[2]] * x_1[:, j]
  end

  series[Symbol("yhat_std_", string(i))] = (yhat - mean(yhat)) / std(yhat);
end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series)), outer = size(chain1_ave[(burnIn + 1):stepsize:end, :])[1]),
  var = Array(stack(series[:, 3:end])[1]),
  val = Array(stack(series[:, 3:end])[2])
  );

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"black")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )

writetable(joinpath(ou_Rdir, "mh_train_out.csv"), series)

# Testing Data
yhat = est1[1] + est1[2] * y_1test
for i in 1:(size(x)[2])
  yhat += est1[i + 2] * x_test[:, i] + est1[i + 2 + size(x)[2]] * x_1test[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)

rmse_ch1 = rmse(y_test, yhat_std)

series_test = DataFrame(Y = Array(indicators[:, 1]));
series_test[:yhat_std] = [y; yhat_std];

plot(layer(series_test, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series_test, y = :yhat_std, Geom.line, style(default_color = colorant"red"))
  )

for i in (burnIn + 1):stepsize:size(chain1_ave)[1]
  yhat = chain1_ave[i, 1] + chain1_ave[i, 2] * y_1test
  for j in 1:(size(x)[2])
    yhat += chain1_ave[i, j + 2] * x_test[:, j] + chain1_ave[i, j + 2 + size(x)[2]] * x_1test[:, j]
  end

  series_test[Symbol("yhat_std_", string(i))] = [y; (yhat - mean(yhat)) / std(yhat)];
end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series_test)), outer = size(chain1_ave[(burnIn + 1):stepsize:end, :])[1]),
  var = Array(stack(series_test[:, 3:end])[1]),
  val = Array(stack(series_test[:, 3:end])[2])
  );

plot(layer(series_test, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series_test, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"black")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )

writetable(joinpath(ou_Rdir, "mh_test_out.csv"), series_test)

"""
Do Bayesian Estimation using Hamiltonian Monte Carlo
"""
srand(123);
U(theta::Array{Float64}) = - logpost(theta)
K(p::Array{Float64}; Σ = eye(length(p))) = (p' * inv(Σ) * p) / 2
function dU(theta::Array{Float64})
  yhat = theta[1] + theta[2] * y_1
  for i in 1:(size(x)[2])
    yhat += theta[i + 2] * x[:, i] + theta[i + 2 + size(x)[2]] * x_1[:, i]
  end
  vcat(
      [-alpha * sum((y - yhat)) + s[1] * theta[1]],
      [-alpha * sum((y - yhat) .* y_1) + s[1] * theta[1]],
      [-alpha * sum((y - yhat) .* x[:, i]) + s[i + 1] * theta[i + 1] for i in 1:size(x)[2]],
      [-alpha * sum((y - yhat) .* x_1[:, i]) + s[i + 2 + size(x)[2]] * theta[i + 2 + size(x)[2]] for i in 1:size(x)[2]]
  )
end
dK(p::Array{Float64}; Σ::Array{Float64} = eye(length(p))) = inv(Σ) * p;

chain2 = Dict{Symbol, Array{Float64}}()

srand(123);
for i in 1:n_chain
  init_val = rand(Uniform(.001 + (i - 1) / 4, i / 4), n_params)
  HMC_object2 = HMC(U, K, dU, dK, init_val, n_params);
  chain2[Symbol("Realization_" * string(i))] = mcmc(HMC_object2, leapfrog_params = Dict([:ɛ => .0009, :τ => 20]), r = 100000);
end

writetable(joinpath(ou_Rdir, "chain2.csv"), DataFrame(chain2))
"""
Plot it
"""
series = DataFrame(Y = y);
burnIn = 10;
stepsize = 10;

est2 = mapslices(mean, chain2[(burnIn + 1):stepsize:end, :], [1]);
stde2 = mapslices(std, chain2[(burnIn + 1):stepsize:end, :], [1]);

# Training Data
yhat = est2[1] + est2[2] * y_1
for j in 1:(size(x)[2])
  yhat += est2[j + 2] * x[:, j] + est2[j + 2 + size(x)[2]] * x_1[:, j]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)

series[:yhat_std] = yhat_std;

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red"))
  )

for i in (burnIn + 1):stepsize:size(chain2)[1]
  yhat = chain2[i, 1] + chain2[i, 2] * y_1
  for j in 1:(size(x)[2])
    yhat += chain2[i, j + 2] * x[:, j] + chain2[i, j + 2 + size(x)[2]] * x_1[:, j]
  end

  series[Symbol("yhat_std_", string(i))] = (yhat - mean(yhat)) / std(yhat);
end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series)), outer = size(chain2[(burnIn + 1):stepsize:end, :])[1]),
  var = Array(stack(series[:, 3:end])[1]),
  val = Array(stack(series[:, 3:end])[2])
  );

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"black")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )
writetable(joinpath(ou_Rdir, "hmc_train_out.csv"), series)

# Testing Data
burnIn = 10;
stepsize = 10;

yhat = est2[1] + est2[2] * y_1test
for i in 1:(size(x)[2])
  yhat += est2[i + 2] * x_test[:, i] + est2[i + 2 + size(x)[2]] * x_1test[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)
rmse_ch2 = rmse(y_test, yhat_std)

series_test = DataFrame(Y = Array(indicators[:, 1]));
series_test[:yhat_std] = [y; yhat_std];

plot(layer(series_test, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series_test, y = :yhat_std, Geom.line, style(default_color = colorant"red"))
  )

for i in (burnIn + 1):stepsize:size(chain1)[1]
  yhat = chain2[i, 1] + chain2[i, 2] * y_1test
  for j in 1:(size(x)[2])
    yhat += chain2[i, j + 2] * x_test[:, j] + chain2[i, j + 2 + size(x)[2]] * x_1test[:, j]
  end

  series_test[Symbol("yhat_std_", string(i))] = [y; (yhat - mean(yhat)) / std(yhat)];
end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series_test)), outer = size(chain1[(burnIn + 1):stepsize:end, :])[1]),
  var = Array(stack(series_test[:, 3:end])[1]),
  val = Array(stack(series_test[:, 3:end])[2])
  );

plot(layer(series_test, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series_test, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"black")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )

writetable(joinpath(ou_Rdir, "hmc_test_out.csv"), series_test)
"""
Do Bayesian Estimation using Stochastic Gradient Hamiltonian Monte Carlo
"""
srand(123);
function dU_noise(theta::Array{Float64})
  yhat = theta[1] + theta[2] * y_1
  for i in 1:(size(x)[2])
    yhat += theta[i + 2] * x[:, i] + theta[i + 2 + size(x)[2]] * x_1[:, i]
  end
  vcat(
      [-alpha * sum((y - yhat)) + s[1] * theta[1]],
      [-alpha * sum((y - yhat) .* y_1) + s[1] * theta[1]],
      [-alpha * sum((y - yhat) .* x[:, i]) + s[i + 1] * theta[i + 1] for i in 1:size(x)[2]],
      [-alpha * sum((y - yhat) .* x_1[:, i]) + s[i + 2 + size(x)[2]] * theta[i + 2 + size(x)[2]] for i in 1:size(x)[2]]
  ) + randn(n_params)
end

SGHMC_object = SGHMC(dU_noise, dK, eye(n_params), eye(n_params), eye(n_params), zeros(n_params), n_params * 1.);

@time chain3 = mcmc(SGHMC_object, leapfrog_params = Dict([:ɛ => .0009, :τ => 20]), r = 100000);

writetable(joinpath(ou_Rdir, "chain3.csv"), DataFrame(chain3))
"""
Plot it
"""
# Training Data
series = DataFrame(Y = y);
burnIn = 10;
stepsize = 10;
est3 = mapslices(mean, chain3[(burnIn + 1):stepsize:end, :], [1]);
stde3 = mapslices(std, chain3[(burnIn + 1):stepsize:end, :], [1]);
print(DataFrame(a = round(vec(est3), 3), b = round(vec(stde3), 3)))

yhat = est3[1] + est3[2] * y_1
for i in 1:(size(x)[2])
  yhat += est3[i + 2] * x[:, i] + est3[i + 2 + size(x)[2]] * x_1[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)

series[:yhat_std] = yhat_std;

plot(
  layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
  layer(series, y = :Y, Geom.line, style(default_color = colorant"orange"))
  )

for i in (burnIn + 1):stepsize:size(chain3)[1]
  yhat = chain3[i, 1] + chain3[i, 2] * y_1
  for j in 1:(size(x)[2])
    yhat += chain3[i, j + 2] * x[:, j] + chain3[i, j + 2 + size(x)[2]] * x_1[:, j]
  end

  series[Symbol("yhat_std_", string(i))] = (yhat - mean(yhat)) / std(yhat);
end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series)), outer = size(chain3[(burnIn + 1):stepsize:end, :])[1]),
  var = Array(stack(series[:, 3:end])[1]),
  val = Array(stack(series[:, 3:end])[2])
  );

plot(layer(series, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"black")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )

writetable(joinpath(ou_Rdir, "sghmc_train_out.csv"), series)

# Testing Data
stepsize = 10;
est3 = mapslices(mean, chain3[1:stepsize:end, :], [1]);

yhat = est3[1] + est3[2] * y_1test
for i in 1:(size(x)[2])
  yhat += est3[i + 2] * x_test[:, i] + est3[i + 2 + size(x)[2]] * x_1test[:, i]
end

yhat_std = (yhat - mean(yhat)) / std(yhat)
rmse_ch3 = rmse(y_test, yhat_std)

series_test = DataFrame(Y = Array(indicators[:, 1]));
plot(series_test, y = :Y, Geom.line, Guide.ylabel("Reference Series"))

series_test[:yhat_std] = [y; yhat_std];

plot(layer(series_test, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series_test, y = :yhat_std, Geom.line, style(default_color = colorant"red"))
  )

for i in (burnIn + 1):stepsize:size(chain3)[1]
  yhat = chain3[i, 1] + chain3[i, 2] * y_1test
  for j in 1:(size(x)[2])
    yhat += chain3[i, j + 2] * x_test[:, j] + chain3[i, j + 2 + size(x)[2]] * x_1test[:, j]
  end

  series_test[Symbol("yhat_std_", string(i))] = [y; (yhat - mean(yhat)) / std(yhat)];
end

series_stacked = DataFrame(
  x = repeat(collect(1:nrow(series_test)), outer = size(chain3[(burnIn + 1):stepsize:end, :])[1]),
  var = Array(stack(series_test[:, 3:end])[1]),
  val = Array(stack(series_test[:, 3:end])[2])
  );

plot(layer(series_test, y = :Y, Geom.line, style(default_color = colorant"orange")),
     layer(series_test, y = :yhat_std, Geom.line, style(default_color = colorant"red")),
     layer(series_stacked, x = :x, y = :val, group = :var, Geom.line, style(default_color = colorant"black")),
     Guide.xlabel("Time"), Guide.ylabel("CLEI and Reference Series")
  )

writetable(joinpath(ou_Rdir, "sghmc_test_out.csv"), series_test)


writetable(joinpath(ou_Rdir, "coefs.csv"), DataFrame([round(est1, 3); round(stde1, 3); round(est2, 3); round(stde2, 3); round(est3, 3); round(stde3, 3)]'));

print(round([rmse_ch1 rmse_ch2 rmse_ch3], 3))
