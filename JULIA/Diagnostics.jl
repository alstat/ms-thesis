"""
Cumulative quantile of the series
"""
function cum_quantile(chain::Array{Float64, 1}, probs::Array{Float64, 1})
  cquant = zeros(length(chain), length(probs))

  for i in 1:length(chain)
    cquant[i, :] = quantile(chain[1:i], probs)
  end

  cquant_df = DataFrame(cquant)
  names!(cquant_df, [Symbol(string(i * 100) * "%") for i in probs])

  return cquant_df
end

"""
BURN-IN DIAGNOSTICS

1. Geweke Diagnostics
      This diagnostic compares the location of the sampled parameter on two different time intervals of the chain.
      If the mean values of the parameter in the two time intervals are somewhat close to each other we can
      assume that the two different parts of the chain have similar locations in the state space, and it is assumed
      that the two samples come from the same distribution. Usually one compares the last half of the chain,
      which is assumed to have converged (in order for the test to make sense), against some smaller interval in
      the beginning of the chain.
"""
function spectrum_ar(x::Array{Real})
  if length(size(x)) > 1
    v0 = order = size(x)[2]
  else
    v0 = order = 1
  end

  z = 1:size(x)[1]

  for i in 1:order
    xz_df = DataFrame(X = x[:, i], Z = z)
    lm_out = fit(LinearModel, X ~ Z, xz_df)
    if



    x <- as.matrix(x)
    v0 <- order <- numeric(ncol(x))
    names(v0) <- names(order) <- colnames(x)
    z <- 1:nrow(x)
    for (i in 1:ncol(x)) {
        lm.out <- lm(x[, i] ~ z)
        if (identical(all.equal(sd(residuals(lm.out)), 0), TRUE)) {
            v0[i] <- 0
            order[i] <- 0
        }
        else {
            ar.out <- ar(x[, i], aic = TRUE)
            v0[i] <- ar.out$var.pred/(1 - sum(ar.out$ar))^2
            order[i] <- ar.out$order
        }
    }
    return(list(spec = v0, order = order))
}

function geweke_diag{T <: Real}(x::Array{T}; xfirst::Real = .1, xlast::Real = .5)
  if !(0.0 < first < 1.0)
    throw(ArgumentError("first is not in (0, 1)"))
  elseif !(0.0 < last < 1.0)
    throw(ArgumentError("last is not in (0, 1)"))
  elseif first + last > 1.0
    throw(ArgumentError("first and last proportions overlap"))
  end

  x = Array(x)
  n = size(x)[1]
  x1 = x[1:round(Int, xfirst * n), :]
  x2 = x[round(Int, n - xlast * n + 1) : n, :]

  x1_mean, x2_mean = mean(x1), mean(x2)
  x1_var, x2_var = spectrum_ar(x1), spectrum_ar(x2)


  maps



end
function (x, frac1 = 0.1, frac2 = 0.5)
{
    if (frac1 < 0 || frac1 > 1) {
        stop("frac1 invalid")
    }
    if (frac2 < 0 || frac2 > 1) {
        stop("frac2 invalid")
    }
    if (frac1 + frac2 > 1) {
        stop("start and end sequences are overlapping")
    }
    if (is.mcmc.list(x)) {
        return(lapply(x, geweke.diag, frac1, frac2))
    }
    x <- as.mcmc(x)
    xstart <- c(start(x), floor(end(x) - frac2 * (end(x) - start(x))))
    xend <- c(ceiling(start(x) + frac1 * (end(x) - start(x))),
        end(x))
    y.variance <- y.mean <- vector("list", 2)
    for (i in 1:2) {
        y <- window(x, start = xstart[i], end = xend[i])
        y.mean[[i]] <- apply(as.matrix(y), 2, mean)
        y.variance[[i]] <- spectrum0.ar(y)$spec/niter(y)
    }
    z <- (y.mean[[1]] - y.mean[[2]])/sqrt(y.variance[[1]] + y.variance[[2]])
    out <- list(z = z, frac = c(frac1, frac2))
    class(out) <- "geweke.diag"
    return(out)
}

function geweke_diag{T<:Real}(x::Vector{T}; first::Real=0.1, last::Real=0.5,
                              etype=:imse, args...)
  if !(0.0 < first < 1.0)
    throw(ArgumentError("first is not in (0, 1)"))
  elseif !(0.0 < last < 1.0)
    throw(ArgumentError("last is not in (0, 1)"))
  elseif first + last > 1.0
    throw(ArgumentError("first and last proportions overlap"))
  end
  n = length(x)
  x1 = x[1:round(Int, first * n)]
  x2 = x[round(Int, n - last * n + 1):n]
  z = (mean(x1) - mean(x2)) /
      sqrt(mcse(x1, etype; args...)^2 + mcse(x2, etype; args...)^2)
  [round(z, 3), round(1.0 - erf(abs(z) / sqrt(2.0)), 4)]
end
