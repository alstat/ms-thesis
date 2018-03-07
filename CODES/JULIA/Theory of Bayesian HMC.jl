"""
The following codes are meant to supplement the chapter 2.
"""
workspace()

using PyPlot
using Distributions
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
  xyplot(:(y1 ~ x), linestyle = "dashed", marker = "", xlab = "x", ylab = "Posterior + Approximator", save = false)
  xyplot(:(y2 ~ x), linestyle = "solid", marker = "", add = true, file_name = joinpath(homedir(), "Dropbox/MS Thesis/Julia Files/Figures", string("Chi_Post_Appr_", i, ".png")))
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

xyplot(:(area ~ draws), marker = "", linestyle = "solid")

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

xyplot(:(x ~ 1:r), marker = "", linestyle = "-", xlab = "Iterations", ylab = "Mixing")

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

xyplot(:(x[(burnIn + 1):r, 2] ~ x[(burnIn + 1):r, 1]), marker = "o", linestyle = "")
