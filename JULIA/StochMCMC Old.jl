module StochMCMC

using Distributions
using DataFrames
using MixedModels

include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/MH.jl"))
include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/HMC.jl"))
include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/SG HMC.jl"))
#include(joinpath(homedir(), "Dropbox/MS THESIS/JULIA/Diagnostics.jl"))

export mcmc, MH, HMC, SGHMC, cum_quantile
end
