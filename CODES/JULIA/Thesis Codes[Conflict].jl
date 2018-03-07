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

# Change working directory
cd(joinpath(homedir(), "Google Drive/MS Thesis/Julia Files"))

# F U N C T I O N S   F O R   P L O T T I N G
using PyPlot
matplotlib[:rc]("font", family = "Times New Roman")
matplotlib[:rc]("xtick", direction = "out")
matplotlib[:rc]("ytick", direction = "out")
locator_params(axis = "x", nbins = 8, tight = true)
locator_params(axis = "y", nbins = 8, tight = true)
tick_params(axis = "x", pad = 15)

# Define colors for plotting"#EFDECD","#3B444B"
col, bgcol, grdcol, lty, ticksize = ("black", "white", "#696969", "dotted", 9)

function graph(x, y = Union{}; xlab = "", ylab = "", color = col, marker = "", markersize = 5,
    ptype = "xyplot", where = "post", width = 1, bins = 30, save = true, filename = "current", filetype = "png")

    if (y == Union{}) & (ptype != "hist")
        y = x
        ylen = y |> length
        x = collect(1:ylen)
    elseif (y == Union{}) & (ptype == "hist")
        y = x
    end

    if save == true
        ioff()
    end

    f, ax = subplots(figsize = (5, 5))

    if ptype == "step"
        ax[:step](x, y, where = where, color = color, marker = marker, markersize = markersize)
    elseif ptype == "xyplot"
        ax[:plot](x, y, color = color, marker = marker, markersize = markersize)
    elseif ptype == "bar"
        ax[:bar](x, y, width, color = color)
    elseif ptype == "hist"
        ax[:hist](x, bins, edgecolor = "white", facecolor = "black")
    end

    ax[:set_xlabel](xlab)
    ax[:set_ylabel](ylab)
    ax[:grid]("on", which = "major", color = grdcol, linestyle = lty)
    ax[:set_axis_bgcolor](bgcol)
    ax[:tick_params](axis = "both", which = "major", labelsize = ticksize, pad = 5)
    ax[:tick_params](axis = "both", which = "minor", labelsize = ticksize, pad = 5)
    ax[:set_axisbelow]("on")
    ax[:margins](.03)

    if save == true
        f[:tight_layout]()
        f[:savefig](filename * "." * filetype)
    end
end
#------------------------------

# M O N T E   C A R L O   S I M U L A T I O N
using Distributions
using StatsBase

srand(123)
draws = collect(1000:100:100000)
area = draws |> length |> zeros

for i in 1:(draws |> length)
    samples = rand(Normal(), draws[i])
    area[i] = sum((samples .> -1.96) & (samples .< 1.96)) / (samples |> length)
end

graph(area, xlab = "Draws", ylab = "Area")

r = 100000
x = r |> zeros
x[1] = 30

for i in 1:r - 1
    propose = x[i] + rand(Uniform(-1, 1))
    accept = rand(Uniform()) < pdf(Cauchy(), propose) / pdf(Cauchy(), x[i])

    if accept == true
        x[i + 1] = propose
    else
        x[i + 1] = x[i]
    end
end

graph(x[1000:end], xlab = "Iterations", ylab = "Mixing")

function BvCauchy(x, μ = [0; 0], γ = 1)
  (1 / (2pi)) * (γ / ((x[1] - μ[1])^2 + (x[2] - μ[2])^2 + γ^2)^1.5)
end

x = zeros((r, 2))
x[1, :] = [-100; 100]

for i in 1:r - 1
    propose = x[i, :]' + rand(Uniform(-5, 5), 2)
    accept = rand(Uniform(), 1)[1] < (BvCauchy(propose) / BvCauchy(x[i, :]))

    if accept == true
        x[i + 1, :] = propose
    else
        x[i + 1, :] = x[i, :]
    end
end

graph(x[1000:end, 1], x[1000:end, 2], xlab = L"$x_1$", ylab = L"$x_2$")
graph(autocor(x[1000:end, 1]), xlab = "Lags", ylab = "Autocorrelation", ptype = "bar")
graph(autocor(x[1000:end, 2]), xlab = "Lags", ylab = "Autocorrelation", ptype = "bar")

function bv_norm(x, μ1 = 10, μ2 = -10, σ1 = 1.5, σ2 = 1.35, ρ = .5)
    1 / (2*pi*σ1*σ2*sqrt(1-ρ^2)) * exp(-(1 / (2*(1 - ρ^2))) * ((((x[1, 1] - μ1)^2) / (σ1^2)) +
    (((x[2, 1] - μ2)^2) / σ2^2) - ((2*ρ*(x[1, 1] - μ1)*(x[2, 1] - μ2)) / (σ1*σ2))))
end

x = zeros((r, 2))
x[1, :] = [0 0]

for i in 1:r - 1
  proposal = x[i, :]' + rand(Uniform(-5, 5), 2)
  accept = rand(Uniform(), 1)[1] < (bv_norm(proposal) / bv_norm(x[i, :]'))

  if accept == true
    x[i + 1, :] = proposal
  else
    x[i + 1, :] = x[i, :]
  end

end

graph(x[1:2000, 1], x[1:2000, 2], xlab = L"$x_1$", ylab = L"$x_2$")

# Gibbs Sampling
function con_norm(x, μ1 = 10, μ2 = -10, σ1 = 1.5, σ2 = 1.35, ρ = .5)
    rand(Normal(μ1 + (σ1 / σ2) * ρ * (x - μ2), sqrt((1 - ρ^2) * σ1^2)), 1)
end

x = zeros((r, 2))

for i in 1:r - 1
    x[i + 1, 1] = con_norm(x[i, 2])[1]
    x[i + 1, 2] = con_norm(x[i + 1, 1], -10, 10, 1.35, 1.5)[1]
end
graph(x[1000:end, 1], x[1000:end, 2], marker = "o", xlab = L"$x_1$", ylab = L"$x_2$", ptype = "step")
graph(autocor(x[1000:end, 1]), xlab = "Lags", ylab = "Autocorrelation", ptype = "bar")
graph(autocor(x[1000:end, 2]), xlab = "Lags", ylab = "Autocorrelation", ptype = "bar")
