"""
DIAGNOSTICS:
"""
include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/StochMCMC.jl"));

using DataFrames: DataFrame, names!, stack
using Distributions
using Plots
using StochMCMC
using Mamba

theme(:dark)
"""
COMPUTING CUMULATIVE QUANTILE PLOT
"""
x = rand(Normal(), 10000);
pr = [.025; .5; .975];
z = cum_quantile(x, pr);
x_df = stack(z[[1; 3]]);
x_df[:iter] = repeat(collect(1:length(x)), outer = 2);
z[:iter] = collect(1:length(x))

p1 = plot(
  layer(x_df, x = :iter, y = :value, group = :variable, Geom.line()),
  layer(z, x = :iter, y = Symbol("50.0%"), Geom.line(), style(default_color = colorant"orange")),
  Guide.xlabel("Iteration"),
  Guide.ylabel("Cumulative Quantile")
);

gewekediag(x) |> showall
heideldiag(x) |> showall
rafterydiag(x) |> showall
p1

using RDatasets
iris = dataset("datasets", "iris");

# load the StatPlots recipes (for DataFrames) available via:
Pkg.add("StatPlots")
using StatPlots
pyplot()
# Scatter plot with some custom settings
scatter(iris, :SepalLength, :SepalWidth, group=:Species,
        title = "My awesome plot",
        xlabel = "Length", ylabel = "Width",
        m=(0.5, [:cross :hex :star7], 12),
        bg=RGB(.2,.2,.2))
