set.seed(123)

# Set the parameters
w0 <- -.3; w1 <- -.5; stdev <- 5

# Define Data Parameters
alpha <- 1 / stdev  # for likelihood

# Generate Hypothetical Data
x <- runif(20, -1, 1)
A <- 1 %>% rep(times = 20) %>% cbind(x)
B <- rbind(w0, w1)
f <- A %*% B 
y <- f + rnorm(20, sd = alpha)

# Define Hyperparameters
Imat <- 2 %>% diag # for prior
b <- 2 #for prior
b1 <- (1 / b)**2 # Square this since in R, rnorm uses standard dev

mu <- 0 %>% rep(times = 2) %>% matrix(ncol = 1) # for prior
s <- b1 * Imat # for prior

# Define log prior
logprior <- function (theta, mu = mu, s = s) {
  w0_prior <- dnorm(theta[1], mu[1,1], s[1,1], log = TRUE)
  w1_prior <- dnorm(theta[2], mu[2,1], s[2,2], log = TRUE)
  w_prior <- c(w0_prior, w1_prior)
   
  w_prior %>% sum %>% return
}

# Define log likelihood
loglike <- function (theta, alpha = alpha, x = x, y = y) {
  yhat <- theta[1] + theta[2]*x
  
  likhood <- numeric()
  for (i in 1:length(yhat)) {
    likhood[i] <- dnorm(y[i], yhat[i], alpha)
  }
  
  return (likhood %>% sum)
}

# Define log posterior
logpost <- function (theta) {
  loglike(theta, alpha = alpha, x = x, y = y) + logprior(theta, mu, s)
}

# Proposal
sigmas <- c(1, 1)
proposal <- function (theta) {
  random <- numeric()
  
  for (i in 1:length(theta)) {
    random[i] <- rnorm(1, theta[i], sigmas[i])
  }
  
  random %>% return
}
 
# Metropolis-Hasting MCMC
mh <- function (theta_start, max_iter) {
  chain <- matrix(NA, max_iter + 1, theta_start %>% length)
  chain[1, ] <- theta_start
  
  for (i in 1:max_iter) {
    propose <- proposal(chain[i, ])
    probab <- exp(logpost(propose) - logpost(chain[i, ]))
    
    if (runif(1) < probab){
      chain[i + 1, ] <- propose
    } else {
      chain[i + 1, ] <- chain[i, ] 
    }
  }
  
  chain %>% return
}

mcmc <- mh(c(0, 0), 10000)
w_est <- mcmc %>% colMeans



histogram(mcmc[, 1], col = "gray50", border = "white") %>%
  update(xlab = expression(paste("Chain Values of ", w[0]))) %>%
  update(panel = function (x, ...) {
    panel.grid(-1, -1)
    panel.histogram(x, ...)
    panel.abline(v = w0, lty = 2, col = "black", lwd = 2)
  })
histogram(mcmc[, 2], col = "gray50", border = "white") %>%
  update(xlab = expression(paste("Chain Values of ", w[1]))) %>%
  update(panel = function (x, ...) {
    panel.grid(-1, -1)
    panel.histogram(x, ...)
    panel.abline(v = w1, lty = 2, col = "black", lwd = 2)
  })

xyplot(y ~ x, col = "black", fill = "gray80", cex = 1.2, type = "p", pch = 21) %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    for (i in (3*r/4):r) {
      yhat <- mcmc[i, 1] + mcmc[i, 2] * x
      panel.xyplot(x, yhat, type = "l", col = "gray60")  
    }
    panel.xyplot(x, y, ...)
    panel.xyplot(x, w_est[1] + w_est[2] * x, type = "l", col = "black", lwd = 4, lty = 2)
  })

xyplot(y ~ x, col = "black", cex = 1.2, type = "p") %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    yhat <- w0_est + w1_est * x
    panel.xyplot(x, yhat, type = "l", col = "gray50")  
    panel.xyplot(x, y, ...)
  })

xyplot((w0_est + w1_est * x) ~ x)
xyplot(output[, 1] ~ 1:nrow(output), type = c("g", "l"), col = "gray50", lwd = 1) %>%
  update(xlab = "Iterations", ylab = expression(paste("Chain Values of ", w[0]))) %>%
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    panel.abline(h = w0, col = "black", lty = 2, lwd = 2)
  })

xyplot(output[, 2] ~ 1:nrow(output), type = c("g", "l"), col = "gray50", lwd = 1) %>%
  update(xlab = "Iterations", ylab = expression(paste("Chain Values of ", w[1]))) %>%
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    panel.abline(h = w1, col = "black", lty = 2, lwd = 2)
  })




####

xyplot(y ~ x, col = "black", fill = "gray80", cex = 1.2, type = "p", pch = 21) %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    for (i in (3*r/4):r) {
      yhat <- mcmc[i, 1] + mcmc[i, 2] * x
      panel.xyplot(x, yhat, type = "l", col = "gray60")  
    }
    panel.xyplot(x, y, ...)
    panel.xyplot(x, w_est[1] + w_est[2] * x, type = "l", col = "black", lwd = 4, lty = 2)
  })



xyplot(y ~ x, col = "black", cex = 1.2, type = "p") %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    yhat <- w0_est + w1_est * x
    panel.xyplot(x, yhat, type = "l", col = "gray50")  
    panel.xyplot(x, y, ...)
  })

xyplot((w0_est + w1_est * x) ~ x)

histogram(output[, 1])
histogram(output[, 2])
histogram(output[, 3])

w0_est <- output[, 1] %>% mean
w1_est <- output[, 2] %>% mean
w0_est
w1_est
#output[, 3] %>% mean

histogram(output[, 1], col = "gray50", border = "white") %>%
  update(xlab = expression(paste("Chain Values of ", w[0]))) %>%
  update(panel = function (x, ...) {
    panel.grid(-1, -1)
    panel.histogram(x, ...)
    panel.abline(v = b0, lty = 2, col = "black", lwd = 2)
  })
histogram(output[, 2], col = "gray50", border = "white") %>%
  update(xlab = expression(paste("Chain Values of ", w[1]))) %>%
  update(panel = function (x, ...) {
    panel.grid(-1, -1)
    panel.histogram(x, ...)
    panel.abline(v = b1, lty = 2, col = "black", lwd = 2)
  })

xyplot(y ~ x, col = "black", cex = 1.2, type = "p") %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    for (i in 1:1000) {
      yhat <- output[i, 1] + output[i, 2] * x
      panel.xyplot(x, yhat, type = "l", col = "gray50")  
    }
    panel.xyplot(x, y, ...)
  })

xyplot(y ~ x, col = "black", cex = 1.2, type = "p") %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    yhat <- w0_est + w1_est * x
    panel.xyplot(x, yhat, type = "l", col = "gray50")  
    panel.xyplot(x, y, ...)
  })

xyplot(output[, 1] ~ 1:nrow(output), type = c("g", "l"), col = "gray50", lwd = 1) %>%
  update(xlab = "Iterations", ylab = expression(paste("Chain Values of ", w[0]))) %>%
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    panel.abline(h = b0, col = "black", lty = 2, lwd = 2)
  })

xyplot(output[, 2] ~ 1:nrow(output), type = c("g", "l"), col = "gray50", lwd = 1) %>%
  update(xlab = "Iterations", ylab = expression(paste("Chain Values of ", w[1]))) %>%
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    panel.abline(h = b1, col = "black", lty = 2, lwd = 2)
  })
```

## Example using MCMC:
```{r }
set.seed(123)

# Set the parameters
w0 <- -.3; w1 <- -.5; stdev <- 5

# Define Data Parameters
alpha <- 1 / stdev  # for likelihood

# Generate Hypothetical Data
x <- runif(20, -1, 1)
A <- 1 %>% rep(times = 20) %>% cbind(x)
B <- rbind(w0, w1)
f <- A %*% B 
y <- f + rnorm(20, sd = alpha)

# Define Hyperparameters
Imat <- 2 %>% diag # for prior
b <- 2 #for prior
b1 <- (1 / b)**2 # Square this since in R, rnorm uses standard dev

mu <- 0 %>% rep(times = 2) %>% matrix(ncol = 1) # for prior
s <- b1 * Imat # for prior

xyplot(y ~ x)

# loglikelihood
loglike <- function (w0, w1, alpha, x, y) {
  yhat <- w0 + w1*x
  
  likhood <- numeric()
  for (i in 1:length(yhat)) {
    likhood[i] <- dnorm(y[i], yhat[i], alpha, log = TRUE)
  }
  
  return (like %>% sum)
}

like_profile <- function (x, y, theta) {
  Grid <- expand.grid(b0 = x, b1 = y)
  
  z <- numeric()
  for (i in 1:nrow(Grid)) {
    z[i] <- loglike(Grid$b0[i], Grid$b1[i], theta[[1]], y = theta[[2]], x = theta[[3]])
  }
  
  Grid$z <- z
  return (Grid)
}

x1 <- seq(-1, 1, length.out = 200) # weight space
x2 <- seq(-1, 1, length.out = 200)

theta <- list(stdev, y[1,], x[1])
params <- like_profile(x1, x2, theta)

p12 <- contourplot(z ~ b0 * b1, params, aspect = 'xy')
p12 <- p12 %>% update(xlab = expression(w[0]), ylab = expression(w[1])) %>%
  update(cuts = 10,  col.regions = ramp.col(), region = TRUE, labels = TRUE) %>%
  update(panel = function(x, y, ...) {
    panel.contourplot(x, y, ...)
    panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
  }, colorkey = FALSE)
#params
p12
```

Define the prior
```{r}
prior <- function (theta, mu, s) {
  w0_prior <- dnorm(theta[1], mu[1,1], s[1,1], log = TRUE)
  w1_prior <- dnorm(theta[2], mu[2,1], s[2,2], log = TRUE)
  w_prior <- c(w0_prior, w1_prior)
  #b_prior <- dmvnorm(c(theta[1], theta[2]), mean = mu, sigma = s)
  return (sum(w_prior))
}

posterior <- function (theta, alpha, x, y, mu, s) {
  return (sum(loglike(theta[1], theta[2], alpha, x, y), prior(theta, mu, s)))
}
```

Define the MCMC
```{r}
sigma_propose <- c(1, 1)
proposal <- function (theta) {
  random <- numeric()
  
  j <- 1
  for (i in sigma_propose) { #.1, .5, and .3 are the standard deviation of proposals
    random[j] <- rnorm(1, theta[j], i) # for the three parameters namely w0, w1, and the 
    j <- j + 1                         # variablility stdev
  }
  
  return (random)
}

proposal1 <- function (theta) {
  random <- theta + runif(2, -3, 3)
  
  return (random)
}


# Metropolis-Hasting
mh <- function (start, iter, params) {
  chain <- matrix(NA, nrow = iter + 1, ncol = length(start))
  chain[1, ] <- start
  
  for (i in 1:iter) {
    propose <- proposal1(chain[i, ])
    
    probab <- exp(posterior(propose, params[[1]], params[[2]], params[[3]], params[[4]], params[[5]]) - 
                    posterior(chain[i, ], params[[1]], params[[2]], params[[3]], params[[4]], params[[5]]))
    
    if (runif(1) < probab) {
      chain[i + 1, ] <- propose
    } else {
      chain[i + 1, ] <- chain[i, ]
    }
  }
  
  return (chain)
}

# Hamiltonian Monte Carlo
hamiltonian <- function (start, iter) {
  
}
```

Sample from the posterior distribution
```{r}
set.seed(123)
output <- mh(c(2, 3), 10000, params = list(alpha, x, y, mu, s))
w0_est <- output[, 1] %>% mean
w1_est <- output[, 2] %>% mean
w0_est
w1_est
#output[, 3] %>% mean

histogram(output[, 1], col = "gray50", border = "white") %>%
  update(xlab = expression(paste("Chain Values of ", w[0]))) %>%
  update(panel = function (x, ...) {
    panel.grid(-1, -1)
    panel.histogram(x, ...)
    panel.abline(v = w0, lty = 2, col = "black", lwd = 2)
  })
histogram(output[, 2], col = "gray50", border = "white") %>%
  update(xlab = expression(paste("Chain Values of ", w[1]))) %>%
  update(panel = function (x, ...) {
    panel.grid(-1, -1)
    panel.histogram(x, ...)
    panel.abline(v = w1, lty = 2, col = "black", lwd = 2)
  })

xyplot(y ~ x, col = "black", cex = 1.2, type = "p") %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    for (i in 1:200) {
      yhat <- output[i, 1] + output[i, 2] * x
      panel.xyplot(x, yhat, type = "l", col = "gray50")  
    }
    panel.xyplot(x, y, ...)
  })

xyplot(y ~ x, col = "black", cex = 1.2, type = "p") %>% 
  update(panel = function(x, y, ...) {
    panel.grid(h = -1, v = -1)
    yhat <- w0_est + w1_est * x
    panel.xyplot(x, yhat, type = "l", col = "gray50")  
    panel.xyplot(x, y, ...)
  })

xyplot((w0_est + w1_est * x) ~ x)
xyplot(output[, 1] ~ 1:nrow(output), type = c("g", "l"), col = "gray50", lwd = 1) %>%
  update(xlab = "Iterations", ylab = expression(paste("Chain Values of ", w[0]))) %>%
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    panel.abline(h = w0, col = "black", lty = 2, lwd = 2)
  })

xyplot(output[, 2] ~ 1:nrow(output), type = c("g", "l"), col = "gray50", lwd = 1) %>%
  update(xlab = "Iterations", ylab = expression(paste("Chain Values of ", w[1]))) %>%
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    panel.abline(h = w1, col = "black", lty = 2, lwd = 2)
  })

```
```{r}
set.seed(123)
eps <- .3
tau <- 20

U <- function (theta) {
  -logpost(theta) 
}

dU <- function (theta, alpha, x, y, b) {
  c(
    - alpha * sum(y - (theta[1] + theta[2] * x)),
    - alpha * sum((y - (theta[1] + theta[2] * x)) * x)
  ) + b * theta
}

K <- function (p) {
  (t(p) %*% p) / 2
}

dK <- function (p) {
  p
}

H <- function (theta, p) {
  U(theta) + K(p)
}

chain <- matrix(NA, nrow = iter, ncol = length(start))
chain[1, ] <- c(0, 0)

for (n in 1:(r-1)) {
  theta <- chain[n, ]
  p <- rnorm(length(theta))
  oldE <- H(theta, p)
  
  for (t_idx in 1:tau) {
    p <- p - (eps / 2) * dU(theta, alpha, x, y, b) 
    theta <- theta + eps * dK(p)
    p <- p - (eps / 2) * dU(theta, alpha, x, y, b) 
  }
  
  newE <- H(theta, p)
  dE <- newE - oldE
 
  if (dE < 0) {
    chain[n + 1, ] <- theta
  } else if (runif(1) < exp(-dE)) {
    chain[n + 1, ] <- theta
  } else {
    chain[n + 1, ] <- chain[n, ]
  }

}
chain %>% colMeans(na.rm = TRUE)
chain %>% head

x <- matrix(NA, r, 2)
x[1, ] <- matrix(c(0, 0))

for (n in 1:(r-1)) {
  xNew <- x[n, ]
  p <- rnorm(length(xNew))
  oldE <- H(xNew, p, mu, sigma)
  
  for (t_idx in 1:tau) {
    p <- p - (eps / 2) * dU(xNew, sigma)
    xNew <- xNew + eps * dK(p)
    p <- p - (eps / 2) * dU(xNew, sigma)
  }
  
  newE <- H(xNew, p, mu, sigma)
  dE <- newE - oldE
  
  if (dE < 0) {
    x[n + 1, ] <- xNew
  } else if (runif(1) < exp(-dE)) {
    x[n + 1, ] <- xNew
  } else {
    x[n + 1, ] <- x[n, ]
  }
}
```
 
```{r}
# Likelihood for 5th data
x1 <- seq(-1, 1, length.out = 200) # weight space
x2 <- seq(-1, 1, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
alpha <- (1/.2)**2 # since the sd in Gaussian noise is .2
z <- numeric()

# Likelihood for 5th data
for (i in 1:nrow(Grid)) {
  z[i] <- dnorm(x = y[5,], sum(A[5,] * Grid[i,]), alpha)
}

Grid$z <- z; z <- matrix(z, 200, 200, byrow = FALSE)

p6 <- contourplot(z ~ y1 * y2, Grid, aspect = 'xy') 
p6 <- p6 %>% update(xlab = expression(w[0]), ylab = expression(w[1])) %>%
  update(cuts = 10,  col.regions = ramp.col(), region = TRUE, labels = TRUE, 
         panel = function(x, y, ...) {
           panel.contourplot(x, y, ...)
           panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
         }, colorkey = FALSE)
p6

dat <- read.csv("/Users/al-ahmadgaidasaad/Google Drive/Research Files/LUSET/LUSETool/CropInfo/Barley.csv")
dat
library(ALUES)
BANANASoilCR
