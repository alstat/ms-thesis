# Thesis plot size 9 x 9 in (plot3d)
#lattice 6 x 6 (square); in or 9 x 6 (6:9)
#lattice 9 x 9 (facet)
# lattice 4 x 4 (square) at 3 x 3 matrix.

library(lattice)
library(magrittr)

# NONLINEAR REG
# Michaelis-Menten Model
mm <- function (x, w1, w2) {
  (w1 * x) / (w2 + x)
}

# Error Function
loss_mm <- function (x, y, w1, w2) {
  err <- (mm(x, w1, w2) - y)
  return(sum(err**2))
}

# Gradient of Error Function
loss_mm_prime <- function (x, y, w1, w2) {
  wrt_w1 <- (2 * (mm(x, w1, w2) - y) * (x / (w2 +  x))) %>% sum
  wrt_w2 <- (2 * (mm(x, w1, w2) - y) * (- (w1 * x) / (w2 + x)**2)) %>% sum
  rbind(wrt_w1, wrt_w2)
}

# SGD Error Function
loss_mm_sgd <- function (x, y, w1, w2) {
  err <- (mm(x, w1, w2) - y)**2
  return(err)
}

# SGD of Error Function
loss_mm_prime_sgd <- function (x, y, w1, w2) {
  wrt_w1 <- (2 * (mm(x, w1, w2) - y) * (x / (w2 +  x))) 
  wrt_w2 <- (2 * (mm(x, w1, w2) - y) * (- (w1 * x) / (w2 + x)**2))
  rbind(wrt_w1, wrt_w2)
}

loss_f <- function (x, y, w1, w2) {
  K <- length(x)
  result <- 0
  for (k in 1:K) {
    result <- result + loss_mm_sgd(x[k], y[k], w1, w2)
  }
  result
}

p_old <- rbind(0,0); p_new <- rbind(205,.01)
precision <- .0000000001; eta <- .0000001
err <- px <- py <- pz <- numeric()
px[1] <- p_new[1,]; py[1] <- p_new[2,]; pz[1] <- loss_f(x, y, p_new[1,], p_new[2, ])
dat <- data.frame(x = x, y = y)
max_iters <- 100000; err_in <- numeric();
p <- 2; params <- list(); params[[1]] <- p_old; params[[2]] <- p_new
params
#while ((abs(err_in[p - 1] - err_in[p]) > precision) && (p < max_iters)) {
while ((norm(params[[p - 1]] - params[[p]], 'F') > precision) && (p < max_iters)) {
  random_indx <- sample(1:nrow(dat), nrow(dat), FALSE)
  new_dat <- dat[random_indx,]
  for (k in 2:length(y)) {
    p_old <- p_new
    p_new <- p_old - eta * loss_mm_prime_sgd(new_dat$x[k], new_dat$y[k], p_old[1,], p_old[2,])
    err[k - 1] <- loss_mm_sgd(new_dat$x[k - 1], new_dat$y[k - 1], p_new[1,], p_new[2,])
  }
  err_in[p - 1] <- sum(err) / nrow(y)
  px[p] <- p_new[1,]; py[p] <- p_new[2,]
  pz[p] <- loss_f(x, y, p_new[1,], p_new[2,])
  p <- p + 1
  params[[p]] <- p_new
}
p_new

# Plot the data and the model
x <- Puromycin$conc[Puromycin$state == "treated"]
y <- Puromycin$rate[Puromycin$state == "treated"]

w1 <- 195.80
w2 <- .04841

xyplot(y ~ x, xlab = "Concentration", ylab = "Velocity") %>%
  update(pch = 19, col = "black", type = c("g", "p")) %>% 
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    x1 <- Puromycin$conc[Puromycin$state == "untreated"]
    y1 <- Puromycin$rate[Puromycin$state == "untreated"]
    panel.xyplot(x1, y1, pch = 17, col = "black")
  }, key = list(corner = c(.9, .1),
                text = list(c("Treated", "Untreated")),
                points = list(pch = c(19, 17))
  )
  )

# NLS for treated
fm1 <- nls(rate ~ Vm * conc/(K + conc), data = Puromycin,
           subset = state == "treated",
           start = c(Vm = 55, K = .011), trace = T, control = nls.control(maxiter = 100, tol = 1e-08, minFactor = 1/10000000))

# NLS for untreated
fm2 <- nls(rate ~ Vm * conc/(K + conc), data = Puromycin,
           subset = state == "untreated",
           start = c(Vm = 55, K = .011), trace = T, control = nls.control(maxiter = 100, tol = 1e-08, minFactor = 1/10000000))
w1_t <- (fm1 %>% coef)[1]
w2_t <- (fm1 %>% coef)[2]
w1_u <- (fm2 %>% coef)[1]
w2_u <- (fm2 %>% coef)[2]

xyplot(y ~ x, xlab = "Concentration", ylab = "Velocity") %>%
  update(pch = 19, col = "gray50", type = c("g", "p")) %>% 
  update(panel = function (x, y, ...) {
    panel.xyplot(x, y, ...)
    x1 <- Puromycin$conc[Puromycin$state == "untreated"]
    y1 <- Puromycin$rate[Puromycin$state == "untreated"]
    panel.xyplot(x1, y1, pch = 17, col = "gray50")
    x2 <- seq(0, 1.1, by = .001)
    y2 <- mm(x2, w1_t, w2_t)
    y3 <- mm(x2, w1_u, w2_u)
    panel.xyplot(x2, y2, type = "l", col = "black", lwd = 2, lty = 2)
    panel.xyplot(x2, y3, type = "l", col = "black", lwd = 2, lty = 3)
  }, key = list(corner = c(.9, .1),
                text = list(c("MM for Treated", "MM for Untreated")),
                lines = list(lty = c(2, 3), lwd = 2)
  )
  )

xyplot(y ~ x, xlab = "Concentration", ylab = "Velocity") %>%
  update(pch = 19, col = "gray50", type = c("g", "p")) %>%
  update(panel = function(x, y, ...) {
    panel.xyplot(x, y, ...)
    x1 <- seq(0, 1.1, by = .001)
    y1 <- mm(x1, w1, w2)
    panel.xyplot(x1, y1, type = "l", col = "black", lwd = 2, lty = 2)
  })

# Plot the Error Surface
mmf=function(x,beta,gamma)
{
  return((beta*x)/(x+gamma))
}

ssr=function(beta,gamma,x,y)
{
  r=y-mmf(x,beta,gamma)
  return(sum(r^2))
}
z=function(b,g,x,y)
{
  zurf=matrix(NA,length(b),length(g))
  for (i in seq(along=b)){
    for (j in seq(along=g))
      zurf[i, j] = ssr(b[i],g[j],x,y)
  }
  return(zurf)
}
conc=Puromycin$conc#c(0.197, 0.1385, 0.0678, 0.0417, 0.0272, 0.0145, 0.00976, 0.00816)
rate=Puromycin$rate#c(21.5, 21.0, 19.0, 16.5, 14.5, 11.0, 8.5, 7.0)
beta=seq(100, 240, len = 100)
gamma=seq(0, 0.2,len = 100)
zdata=z(beta,gamma,conc,rate)
zdata
contour(beta,gamma,zdata,cuts=1000,xlab="Beta",ylab="Gamma")
##
# Plot the Error Surface
w1 <- seq(50, 250, length = 15)
w2 <- seq(0, 0.1, length = 15)
z <- matrix(NA, nrow = 15, ncol = 15)
for (i in seq(along = w1)) {
  for (j in seq(along = w2)) {
    z[i, j] <- loss_mm(x, y, w1[i], w2[j])
  }
}

library(plot3D)
par(mai = c(.2, .2, .2, .2))
persp3D(w1, w2, z, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
        border = '#8B8589', alpha = .02, colkey = FALSE, theta = 120, phi = 30,
        xlab = "Maximal Velocity", ylab = "Michaelis Constant", zlab = "Error Sum of Square")
for (i in 2:nrow(trace1)) {
  arrows3D(trace1$x[i - 1], trace1$y[i - 1], trace1$z[i - 1], trace1$x[i], trace1$y[i], trace1$z[i], add = TRUE, col = "black", lwd = 2)
}
points3D(trace1$x[1], trace1$y[1], trace1$z[1], pch = 22, col = "black", bg = "white", add = TRUE)
points3D(trace1$x[-c(1, 12)], trace1$y[-c(1, 12)], trace1$z[-c(1, 12)], pch = 19, add = TRUE, col = "black")
points3D(trace1$x[12], trace1$y[12], trace1$z[12], pch = 17, col = "black", bg = "white", add = TRUE)
rect3D(x0 = 0, y0 = 0, z0 = 0, x1 = 500, z1 = max(z), facets = NA, border = "black", lwd = 2, lty = 2, add = TRUE)

# Plot the Error Contour
w1 <- seq(50, 250, length = 200)
w2 <- seq(0,  0.1, length = 200)
Grid <- expand.grid(y1 = w1, y2 = w2)
z <- numeric()
for (i in 1:(200*200)) {
  z[i] <- loss_mm(x, y, Grid[i, 1], Grid[i, 2])
}
Grid$z_val <- z
contourplot(z_val ~ y1 * y2, Grid, cuts = 30, xlab = "Maximal Velocity", ylab = "Michaelis Constant", col.regions = ramp.col(n = 1000), region = TRUE, labels = FALSE,
          panel = function (x, y, z, ...) {
            panel.levelplot(x, y, z, ...)
            for (i in 1:(nrow(trace1))) {
              panel.arrows(trace1$x[i - 1], trace1$y[i - 1], trace1$x[i], trace1$y[i], lwd = 2, col = "black")
            }
          })

## simplest form of fitting the Michaelis-Menten model to these data
fm1 <- nls(rate ~ Vm * conc/(K + conc), data = Puromycin,
           subset = state == "treated",
           start = c(Vm = 55, K = .011), trace = T, control = nls.control(maxiter = 100, tol = 1e-08, minFactor = 1/10000000))
m <- nlsContourRSS(fm1)
plot(m, col = FALSE, nlev = 100)
trace1 <- read.table(text = "
z x y
127395.8   55.000  0.011
6232.493   192.44325192   0.07922019
                     1220.851   213.03856637   0.06199833
                     1195.547   212.53934324   0.06387373
                     1195.45   212.66825917   0.06409704
                     1195.449   212.68224250   0.06411894
                     1195.449   212.68359854   0.06412106
                     1195.449   212.68372881   0.06412126
                     1195.449   212.68374157   0.06412128
                     1195.449   212.68374270   0.06412128
                     1195.449   212.68374323   0.06412128
                     1195.449   212.68374322   0.06412128
                     1195.449   212.68374322   0.06412128", header = TRUE)
trace1
fm1 %>% traceOn
fm1$m
# Use BGD
R <- 1; PRECISION <- .0000000001; ETA <- .0000001
beta_old <- rbind(0, 0); beta_new <- rbind(205, 0.01)
px <- py <- pz <- numeric()
px[1] <- beta_new[1,]; py[1] <- beta_new[2,]; pz[1] <- loss_mm(x, y, beta_new[1,], beta_new[2,])
while (norm(beta_old - beta_new, 'F') > PRECISION) {
  beta_old <- beta_new
  beta_new <- beta_old - ETA * loss_mm_prime(x, y, beta_old[1,], beta_old[2,])
  R <- R + 1
  px[R] <- beta_new[1,]; py[R] <- beta_new[2,]; pz[R] <- loss_mm(x, y, beta_new[1,], beta_new[2,])
}




R
xx <- 1:length(pz)
beta_new
px
xyplot(log(pz) ~ xx, type = "l")
set.seed(12345)
x <- rnorm(100, 15, 2); beta <- rbind(3.4, .75)
err <- rnorm(100); X <- cbind(1, x)
y <- X %*% beta + err

library(lattice)
xyplot(y ~ x, xlab = 'x', ylab = 'y', 
       type = c('p', 'g'), cex = 1, col = 'black')

lm(y ~ x)

loss_mm_prime <- function (y, X, beta) {
  
}

loss_prime <- function (y, X, beta) {
  error <- X %*% beta - y
  denom <- nrow(y)
  (t(X) %*% error) / denom
}

# BGD
R <- 1; PRECISION <- .0001; ETA <- .002
beta_old <- rbind(0, 0); beta_new <- rbind(-3, -3)
px <- py <- pz <- numeric()
px[1] <- beta_new[1,]; py[1] <- beta_new[2,]; pz[1] <- loss(y, X, beta_new)
while (norm(beta_old - beta_new, 'F') > PRECISION) {
  beta_old <- beta_new
  beta_new <- beta_old - ETA * loss_prime(y, X, beta_old)
  R <- R + 1
  px[R] <- beta_new[1,]; py[R] <- beta_new[2,]; pz[R] <- loss(y, X, beta_new)
}
R; beta_new
R

z_val1 <- numeric(); beta0 <- beta1 <- seq(-5, 5, length.out = 15)
g <- expand.grid(b0 = beta0, b1 = beta1); beta <- t(as.matrix(g))
for (i in 1:ncol(beta)) {
  z_val1[i] <- loss(y, X, cbind(beta[, i]))
}
z_val1 <- matrix(z_val1, 15, 15)

library(plot3D)
par(mai = c(.2, .2, .2, .2))
#persp3D(beta0, beta1, z_val1, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
#        border = '#8B8589', alpha = .1, colkey = FALSE, theta = 120,
#        xlab = '', ylab = '')
persp3D(beta0, beta1, z_val1, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
        border = '#8B8589', alpha = .1, colkey = FALSE, theta = 120,
        xlab = '', ylab = '')

points3D(px[1], py[1], pz[1], add = TRUE, bg = 'white', col = 'black', pch = 22, cex = 1.5, colkey = FALSE)
points3D(px[2:(R-1)], py[2:(R-1)], pz[2:(R-1)], col = 'black', add = TRUE, cex = 1, pch = 19)
points3D(px[R], py[R], pz[R], bg = 'white', col = 'black', add = TRUE, cex = 2.5, pch = 24)

text3D(1, 5.6, 0, labels = expression(beta[0]), add = TRUE)
text3D(6.1, 0, 0, labels = expression(beta[1]), add = TRUE)

xyplot(log(pz) ~ 1:length(pz), xlab = 'No. of Iterations', 
       ylab = expression(paste('log', '(', E['in'], ')', sep = '')), 
       type = c('l', 'g'), lwd = 2, col = 'black')
length(px)
class(py[1])
plot(log(pz), type = 'l')
plot(x1, y)
abline(coef = beta_new)
beta_new



# SGD
set.seed(12345)

x <- cbind(rnorm(100, 15, 2))
beta <- rbind(3.4, .75)
err <- rnorm(100)

X <- cbind(1, x)

y <- X %*% beta + err
lm(y ~ x)

loss <- function (y, x, beta) {
  ((beta[1,] + beta[2,] * x - y)**2) / 2
}

loss_f <- function (y, x, beta) {
  K <- nrow(y)
  result <- 0
  for (k in 1:K) {
    result <- result + loss(y[k], x[k], beta) / K
  }
  result
}

z_val1 <- numeric()
beta0 <- beta1 <- seq(-5, 5, length.out = 15)
#px[p-1]
beta0 <- seq(2.7, 2.755, length.out = 15)
beta1 <- seq(.69, .9, length.out = 15)
g <- expand.grid(b0 = beta0, b1 = beta1)
beta <- t(as.matrix(g))
for (i in 1:ncol(beta)) {
  z_val1[i] <- loss_f(y, x, cbind(beta[, i]))
}
z_val2 <- matrix(z_val1, 15, 15)
z_val2
par(mai = c(.2, .2, .2, .2))
persp3D(beta0, beta1, z_val2, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
        border = '#8B8589', alpha = .1, colkey = FALSE, theta = 120,
        xlab = '', ylab = '')

points3D(px[1], py[1], pz[1], add = TRUE, bg = 'white', col = 'black', pch = 22, cex = 1.5, colkey = FALSE)
points3D(px[2:(p-2)], py[2:(p-2)], pz[2:(p-2)], col = 'black', add = TRUE, cex = 1, pch = 19)
points3D(px[p-1], py[p-1], pz[p-1], bg = 'white', col = 'black', add = TRUE, cex = 2.5, pch = 24)
points3D(coef(lm(y ~ x))[1], coef(lm(y ~ x))[2], loss_f(y, x, cbind(coef(lm(y ~ x)))), bg = 'white', col = 'black', add = TRUE, cex = 2.5, pch = 7)
text3D(1, 5.6, 0, labels = expression(beta[0]), add = TRUE)
text3D(6.1, 0, 0, labels = expression(beta[1]), add = TRUE)


binorm <- function (x, mu_vec, sigma_mat) {
  if (is.numeric(x)) {
    n <- length(x)
    x <- matrix(x)
  } else if (is.matrix(x)) {
    n <- nrow(x)
  } 
  
  if (!is.matrix(mu_vec)) stop('mu_vec should be (n x 1) matrix')
  if (!is.matrix(sigma_mat)) stop('sigma_mat should be (n x n) matrix')
  
  mean_deviation <- x - mu_vec
  sigma_inv <- solve(sigma_mat)
  1 / sqrt(((2 * pi)**n * det(sigma_mat))) * exp((-1 / 2) * t(mean_deviation) %*% sigma_inv %*% mean_deviation)
}

gmm <- function (x, theta, mu_list = list(), sigma_list = list()) {
  B <- length(theta)
  out <- 0
  for (b in 1:B) {
    out <- out + theta[b] * binorm(x, mu_list[[b]], sigma_list[[b]])
  }
  out
}

mu_1 <- matrix(c(0, 0)); mu_2 <- matrix(c(-2, 0)); mu_3 <- matrix(c(2, 2))
sigma_1 <- matrix(c(.7, .2, .2, .5), 2, 2)
sigma_2 <- matrix(c(.6, .15, .15, .2), 2, 2)
sigma_3 <- matrix(c(.8, .15, .15, 1.6), 2, 2)
list_mu <- list(mu_1, mu_2, mu_3)
list_sigma <- list(sigma_1, sigma_2, sigma_3)
theta <-c(1/3, 1/6, 1/2)
sum(theta )
x1 <- seq(-4, 4, length.out = 200)
x2 <- seq(-2, 5, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
head(Grid)
z <- numeric()
for (i in 1:nrow(Grid)) {
  z[i] <- gmm(x = c(Grid[i, 1], Grid[i, 2]), theta, list_mu, list_sigma)
  #z[i] <- binorm(x = c(Grid[i, 1], Grid[i, 2]), mu_1, sigma_1)
}
Grid$z <- z
z <- matrix(z, 200, 200, byrow = FALSE)

library(plot3D)
par(mai = c(.2, .2, .2, .2))
persp3D(x1, x2, z, col = ramp.col(c('gray', 'gray10')), colkey = FALSE, 
        theta = 230, phi = 20, lighting = TRUE, ltheta = 160, lphi = 0,
        xlab = '', ylab = '')#, ticktype = 'detailed')

text3D(-.5, 5.7, 0, labels = expression(y[1]), add = TRUE)
text3D(-4.7, 2.5, 0, labels = expression(y[2]), add = TRUE)

#contour3D(x1, x2, z = 0, colvar = z, add = TRUE, col = 'black', .2)
par(mai = c(.9, .9, .3, .3), family = "serif", cex.lab = 1.5)

levelplot(z ~ y1 * y2, Grid, cuts = 10, col.regions = ramp.col(), region = TRUE, xlab = expression(y[1]), ylab = expression(y[2]))
contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(y[1]), ylab = expression(y[2]), labels = FALSE)
          panel = function(x, y, z, ...) {
            panel.levelplot(x, y, z, ...)
            LOW <- p - 500
            UP <- p - 1
            for (points in LOW:UP) {
              panel.arrows(px[points - 1], py[points - 1], px[points], py[points], col = 'black')
            }
            panel.points(px[p-1], py[p-1], pch = 24, cex = 2, fill = 'white', col = 'black')
            panel.points(c1, c2, pch = 21, cex = 2, fill = 'white', col = 'black')
            #panel.rect(c1 -.005, .7865, c1 +.005, .793, col = gray(.95), alpha = .8)
            panel.text(c1, c2 - .0073, labels = expression(bold('OLS Estimate')), col = 'white')
            #panel.rect(px[points - 1] -.005, .8039, px[points - 1] +.005,.8101, col = gray(.95), alpha = .8)
            panel.text(px[points - 1], py[points - 1] - .0073, labels = expression(bold('SGD Estimate')), col = 'white')
          })
contour2D(z, x1, x2, col = 'black', colkey = FALSE)


# Linear Reg
# 550 x 350
# Simulate
set.seed(12345)
x <- rnorm(100, 15, 2); beta <- rbind(3.4, .75)
err <- rnorm(100); X <- cbind(1, x)
y <- X %*% beta + err

library(lattice)
xyplot(y ~ x, xlab = 'x', ylab = 'y', 
       type = c('p', 'g'), cex = 1, col = 'black')

lm(y ~ x)

loss <- function (y, X, beta) {
  error <- X %*% beta - y
  denom <- 2 * nrow(y)
  sum(error**2) / denom
}

loss_prime <- function (y, X, beta) {
  error <- X %*% beta - y
  denom <- nrow(y)
  (t(X) %*% error) / denom
}

# BGD
R <- 1; PRECISION <- .0001; ETA <- .002
beta_old <- rbind(0, 0); beta_new <- rbind(-3, -3)
px <- py <- pz <- numeric()
px[1] <- beta_new[1,]; py[1] <- beta_new[2,]; pz[1] <- loss(y, X, beta_new)
while (norm(beta_old - beta_new, 'F') > PRECISION) {
  beta_old <- beta_new
  beta_new <- beta_old - ETA * loss_prime(y, X, beta_old)
  R <- R + 1
  px[R] <- beta_new[1,]; py[R] <- beta_new[2,]; pz[R] <- loss(y, X, beta_new)
}
R; beta_new
R

z_val1 <- numeric(); beta0 <- beta1 <- seq(-5, 5, length.out = 15)
g <- expand.grid(b0 = beta0, b1 = beta1); beta <- t(as.matrix(g))
for (i in 1:ncol(beta)) {
  z_val1[i] <- loss(y, X, cbind(beta[, i]))
}
z_val1 <- matrix(z_val1, 15, 15)

library(plot3D)
par(mai = c(.2, .2, .2, .2))
#persp3D(beta0, beta1, z_val1, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
#        border = '#8B8589', alpha = .1, colkey = FALSE, theta = 120,
#        xlab = '', ylab = '')
persp3D(beta0, beta1, z_val1, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
        border = '#8B8589', alpha = .1, colkey = FALSE, theta = 120,
        xlab = '', ylab = '')

points3D(px[1], py[1], pz[1], add = TRUE, bg = 'white', col = 'black', pch = 22, cex = 1.5, colkey = FALSE)
points3D(px[2:(R-1)], py[2:(R-1)], pz[2:(R-1)], col = 'black', add = TRUE, cex = 1, pch = 19)
points3D(px[R], py[R], pz[R], bg = 'white', col = 'black', add = TRUE, cex = 2.5, pch = 24)

text3D(1, 5.6, 0, labels = expression(beta[0]), add = TRUE)
text3D(6.1, 0, 0, labels = expression(beta[1]), add = TRUE)

xyplot(log(pz) ~ 1:length(pz), xlab = 'No. of Iterations', 
       ylab = expression(paste('log', '(', E['in'], ')', sep = '')), 
       type = c('l', 'g'), lwd = 2, col = 'black')
length(px)
class(py[1])
plot(log(pz), type = 'l')
plot(x1, y)
abline(coef = beta_new)
beta_new



# SGD
set.seed(12345)

x <- cbind(rnorm(100, 15, 2))
beta <- rbind(3.4, .75)
err <- rnorm(100)

X <- cbind(1, x)

y <- X %*% beta + err
lm(y ~ x)

loss <- function (y, x, beta) {
  ((beta[1,] + beta[2,] * x - y)**2) / 2
}

loss_f <- function (y, x, beta) {
  K <- nrow(y)
  result <- 0
  for (k in 1:K) {
    result <- result + loss(y[k], x[k], beta) / K
  }
  result
}

z_val1 <- numeric()
beta0 <- beta1 <- seq(-5, 5, length.out = 15)
#px[p-1]
beta0 <- seq(2.7, 2.765, length.out = 15)
beta1 <- seq(.68, .91, length.out = 15)
g <- expand.grid(b0 = beta0, b1 = beta1)
beta <- t(as.matrix(g))
for (i in 1:ncol(beta)) {
  z_val1[i] <- loss_f(y, x, cbind(beta[, i]))
}
z_val2 <- matrix(z_val1, 15, 15)
z_val2
par(mai = c(.2, .2, .2, .2))
persp3D(beta0, beta1, z_val2, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
        border = '#8B8589', alpha = .1, colkey = FALSE, theta = 120,
        xlab = '', ylab = '')

points3D(px[1], py[1], pz[1], add = TRUE, bg = 'white', col = 'black', pch = 22, cex = 1.5, colkey = FALSE)
points3D(px[2:(p-2)], py[2:(p-2)], pz[2:(p-2)], col = 'black', add = TRUE, cex = 1, pch = 19)
points3D(px[p-1], py[p-1], pz[p-1], bg = 'white', col = 'black', add = TRUE, cex = 2.5, pch = 24)
points3D(coef(lm(y ~ x))[1], coef(lm(y ~ x))[2], loss_f(y, x, cbind(coef(lm(y ~ x)))), bg = 'white', col = 'black', add = TRUE, cex = 2.5, pch = 7)
text3D(1, 5.6, 0, labels = expression(beta[0]), add = TRUE)
text3D(6.1, 0, 0, labels = expression(beta[1]), add = TRUE)


par(mai = c(.3, .2, .1, .2))
persp3D(beta0, beta1, z_val2, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.4))), 
        border = '#8B8589', alpha = .1, colkey = FALSE, theta = 90,
        xlab = '', ylab = '')
lines3D(px[(p-500):(p-2)], py[(p-500):(p-2)], pz[(p-500):(p-2)], col = ramp.col(n = 100, col = c(gray(.5), gray(0.1), gray(.05))), add = TRUE, lwd = 2, colkey = FALSE)
text3D(2.763, .8, 0, labels = expression(w[1]), add = TRUE)


coef(
  lm(y ~ x)
  )
p_new
cbind(coef(lm(y ~ x)))
loss_f(y, x, cbind(coef(lm(y ~ x))))
px[p]

beta0 <- seq(2.7, 2.755, length.out = 100)
beta1 <- seq(.69, .9, length.out = 100)


beta0 <- seq(2.7, 2.765, length.out = 100)
beta1 <- seq(.68, .91, length.out = 100)
z_val1 <- numeric()
g <- expand.grid(b0 = beta0, b1 = beta1)
beta <- t(as.matrix(g))
for (i in 1:ncol(beta)) {
  z_val1[i] <- loss_f(y, x, cbind(beta[, i]))
}
z_val2 <- matrix(z_val1, 100, 100)
g$z <- z_val1
c1 <- coef(lm(y ~ x))[1]
c1
c2 <- coef(lm(y ~ x))[2]
levelplot(z ~ b0 * b1, g, col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]),
          panel = function(x, y, z, ...) {
            panel.levelplot(x, y, z, ...)
            LOW <- p - 500
            UP <- p - 1
            for (points in LOW:UP) {
              panel.arrows(px[points - 1], py[points - 1], px[points], py[points], col = 'black')
            }
            panel.points(px[p-1], py[p-1], pch = 24, cex = 2, fill = 'white', col = 'black')
            panel.points(c1, c2, pch = 21, cex = 2, fill = 'white', col = 'black')
            #panel.rect(c1 -.005, .7865, c1 +.005, .793, col = gray(.95), alpha = .8)
            panel.text(c1, c2 + .013, labels = expression(bold('OLS Estimate')), col = 'white')
            #panel.rect(px[points - 1] -.005, .8039, px[points - 1] +.005,.8101, col = gray(.95), alpha = .8)
            panel.text(px[points - 1], py[points - 1] - .013, labels = expression(bold('SGD Estimate')), col = 'white')
          })
plot(coef(lm(y ~ x))[1], coef(lm(y ~ x))[2], pch = 24, cex = 2, bg = 'white', col = 'black', xlim = c(2.7, 2.74), ylim = c(.73, .87))
points(px[p-1], py[p-1], pch = 21, cex = 2, bg = 'white', col = 'black')
points3D(px[p], py[p], pz[p], add = TRUE, bg = 'red', col = 'black', pch = 24, cex = 2, colkey = FALSE)
p_new
py[p]
(p-(p-10))
p - 10
params[p]
loss_prime <- function (y, x, beta) {
  b1 <- beta[1,] + beta[2,] * x - y
  b2 <- (beta[1,] + beta[2,] * x - y) * x
  rbind(b1, b2)
}

p_old <- rbind(0,0); p_new <- rbind(-3,-3)
precision <- .0001; eta <- .002
err <- px <- py <- pz <- numeric()
px[1] <- p_new[1,]; py[1] <- p_new[2,]; pz[1] <- loss_f(y, x, p_new)
dat <- data.frame(x = x, y = y)
max_iters <- 10002; err_in <- numeric();
p <- 2; params <- list(); params[[1]] <- p_old; params[[2]] <- p_new
params
#while ((abs(err_in[p - 1] - err_in[p]) > precision) && (p < max_iters)) {
while ((norm(params[[p - 1]] - params[[p]], 'F') > precision) && (p < max_iters)) {
  random_indx <- sample(1:nrow(dat), nrow(dat), FALSE)
  new_dat <- dat[random_indx,]
  for (k in 2:nrow(y)) {
    p_old <- p_new
    p_new <- p_old - eta * loss_prime(new_dat$y[k], new_dat$x[k], p_old)
    err[k - 1] <- loss(new_dat$y[k - 1], new_dat$x[k - 1], p_new)
  }
  err_in[p - 1] <- sum(err) / nrow(y)
  px[p] <- p_new[1,]; py[p] <- p_new[2,]
  pz[p] <- loss_f(y, x, p_new)
  p <- p + 1
  params[[p]] <- p_new
}

p-1
length(err_in[-c(1,2)])
head(px)
p-1
 pz

#p_newsgd <- 
p_new
xyplot(log(err_in) ~ 1:length(err_in), xlab = 'No. of Iterations', ylab = expression(paste('log', '(', E['in'], ')', sep = '')), type = c('l', 'g'), lwd = 2, col = 'black')
points3D(px, py, pz, pch = 19, cex = 1, add = TRUE, col = 'black', colkey = FALSE)

p_new
lines3D(px, py, pz)

py
px
xyplot(err ~ 1:100, type = 'l')
sum(err)
r_history
sample(c(1,2,3), replace = FALSE)
k <- 17
p_new - eta * loss_prime(y[k], x1[k], p_old)

nrow(x1)
x1
err
### 1 var gradient
f <- function (beta) beta**4 - 3 * beta**3 + 2
f_prime <- function (beta) 4 * beta**3 - 9 * beta**2

beta_old <- 0; beta_new <- .1; gamma <- 0.01; precision <- 0.00001
i <- 0; p <- numeric(); p[1] <- beta_new
while (abs(beta_new - beta_old) > precision) {
  beta_old <- beta_new
  beta_new <- beta_old - gamma * f_prime(beta_old)
  i <- i + 1
  p[i + 1] <- beta_new
}

b <- seq(-1, 3, length.out = 100)
library(lattice)
xyplot(f(b) ~ b, col = 'black', lwd = 2, lty = 'dashed', type = c('l', 'g'), ylim = c(-15, 25),
       xlab = 'w', ylab = expression(E['in'](w)),
       panel = function(x, y, ...) {
         panel.xyplot(x, y, ...)
         panel.xyplot(b, f_prime(b), lwd = '2', lty = 'dotted', col = 'black', type = 'l')
         for (i in 1:length(p)) {
           panel.segments(p[i], f(p[i]), p[i], f_prime(p[i]))
         }
       }, key = list(corner = c(.1, .9),
                     text = list(
                       c('Loss Function (LF)', 
                         '',
                         'LF 1st Derivative',
                         '',
                         'Gradient Segments'
                       )
                     ),
                     lines = list(lty = c(2, 0, 3, 0, 1), lwd = 2))
       )


library(plot3D)

# Rosenbrock Function
f1 <- function (x, y) {
  out <- (1 - x) ** 2 + 100 * (y - x**2)**2
  return (out)
}


# Initial Guess
w_old <- matrix(c(0, 0)); w_new <- matrix(c(-1.8,-.8))
gamma <- .0002 # set the learning rate
precision <- .00001 # set the precision

f1_primew1 <- function (w1, w2) {
  out <- -2 * (1 - w1) - 400 * (w2 - w1**2) * w1
  return (out)
}
f1_primew2 <- function (w1, w2) {
  out <- 200 * (w2 - w1**2)
  return (out)
}

# Gradient Vector
g_vec <- function (w1, w2) {
  matrix(c(f1_primew1(w1, w2), f1_primew2(w1, w2)))
}

i <- 1; cx <- cy <- sx <- sy <- sz <- numeric()
cx[1] <- sx[1] <- w_new[1,]
cy[1] <- sy[1] <- w_new[2,]
sz[1] <- f1(sx[1], sy[1])

while(norm(w_new - w_old, 'F') > precision) {
  w_old <- w_new
  w_new <- w_old - gamma *  g_vec(w_old[1,], w_old[2,])
  i <- i + 1
  cx[i] <- sx[i] <- w_new[1, ]; cy[i] <- sy[i] <- w_new[2, ]
  sz[i] <- f1(sx[i], sy[i])
}
i
x <- seq(-2, 2, length.out = 200)
y <- seq(-1, 3, length.out = 200)

g <- expand.grid(x = x, y = y)
g$z <- (1 - g$x) ** 2 + 100 * (g$y - g$x**2)**2
levelplot(z ~ x * y, g, col.regions = ramp.col(), region = TRUE, xlab = expression(w[1]), ylab = expression(w[2]),
          panel = function(x, y, z, ...) {
            panel.levelplot(x, y, z, ...)
            panel.points(cx[1], cy[1], fill = 'white', col = 'black', pch = 22, cex = 1.5)
            panel.points(cx[2:(i-1)], cy[2:(i-1)], bg = 'white', col = 'black', pch = 19, cex = 1)
            panel.points(cx[i], cy[i], fill = 'white', col = 'black', pch = 24, cex = 1.5)
          })

x <- seq(-2, 2, length.out = 15)
y <- seq(-1, 3, length.out = 15)
z <- outer(x, y, f1)
par(mai = c(.2, .2, .2, .05))
persp3D(x, y, z, col = ramp.col(n = 100, col = c(gray(.9), gray(0.5), gray(.6))), 
        border = '#8B8589', alpha = .05, colkey = FALSE, theta = 20, phi = 30,#, ticktype = 'detailed',
        xlab = '', ylab = '', zlab = '')

contour3D(z = 0, x, y, colvar = z, add = TRUE, col = '#B2BEB5')
points3D(sx[1], sy[1], sz[1], add = TRUE, bg = 'white', col = 'black', pch = 22, cex = 1.5, colkey = FALSE)
points3D(sx[2:(i-1)], sy[2:(i-1)], sz[2:(i-1)], col = 'black', add = TRUE, cex = 1, pch = 19)
points3D(sx[i], sy[i], sz[i], bg = 'white', col = 'black', add = TRUE, cex = 2.5, pch = 24)
text3D(0, -1.4, 0, labels = expression(w[1]), add = TRUE)
text3D(2.2, 0.2, 0, labels = expression(w[2]), add = TRUE)
text3D(-2.4,-1.5, 1450, labels = expression(E['in'](bold(w))), add = TRUE)




plot.new()

# KL divergence
library(magrittr); library(lattice)
x <- seq(-5,  8, length.out = 100)
{ dnorm(x) ~ x } %>% xyplot(
  col = 'black', type = 'l', lwd = 2,
  ylab = 'Density',
  panel = function (x, y, ...) {
    panel.grid(h = -1, v = -1)
    panel.xyplot(x, y, ...)
    panel.curve(dnorm(x, 2, 2), -5, 8, lty = 2, lwd = 2)
    panel.arrows(4, dnorm(4, 2, 2), 4.8, .205, angle = 20)
    panel.text(5, .22, labels = expression(q %==% bold(G)(X == x)))
    panel.arrows(-1.5, dnorm(-1.5), -2.8, .235, angle = 20)
    panel.text(-3, .25, labels = expression(p %==% bold(P)(X == x)))
  } #key = list(corner = c(.9, .9),
    #            text = list(c(expression(paste('f(x', '|', theta[1], ')')),
    #                          expression(paste('f(x', '|', theta[0], ')')))),
    #            lines = list(lty = c(1, 2), lwd = 2)
  #  )
)


library(magrittr); library(lattice)
x <- seq(-5,  5, length.out = 100)
{ dnorm(x) * log(dnorm(x)/dnorm(x, 2, 2)) ~ x } %>% xyplot(
  col = 'black', type = 'l', lwd = 2,
  ylab = 'Curve',
  panel = function (x, y, ...) {
    panel.grid(h = -1, v = -1)
    from.z <- -4
    to.z <- 5
    S.x  <- c(from.z, seq(from.z, to.z, 0.01), to.z)
    S.y  <- c(0, dbeta(seq(from.z, to.z, 0.01), 3, 2) / dbeta(seq(from.z, to.z, 0.01), 4, 2), 0)
    S.y  <- c(0, dnorm(seq(from.z, to.z, 0.01)) * log(dnorm(seq(from.z, to.z, 0.01))/dnorm(seq(from.z, to.z, 0.01), 2, 2)), 0)
    panel.polygon(S.x, S.y, col = 'gray', border = 'white')
    panel.xyplot(x, y, ...)
    panel.abline(h = 0, lty = 'dashed', lwd = 2)
    panel.text(2.4, .4, labels = 'Area under this curve')
    panel.text(2.4, .365, labels = expression(paste('is ', D[KL](paste(p, '||', q)))))
    panel.arrows(-.3, .16, 1.85, .34, angle = 20)
  })






kl <- function (x, m1 = 0, s1 = 1, m2 = 2, s2 = 2) {
  (log((s2**2) / (s1**2)) - 1 + (((s1**2) + ((m1 - m2)**2)) / (s2**2))) / 2
}
kl()



library(ggplot2)
f <- function (beta) beta**4 - 3 * beta**3 + 2

beta <- seq(-2, 4, length.out = 100)
qplot(beta, f(beta), xlab = expression(beta), ylab = expression(f(beta)), geom = 'line') + 
  geom_vline(xintercept = 9/4, col = 'red')


#####
library(alues)


p <- function (z) {
  exp(- (z**2) / 2) / (1 + exp(-(20 * z + 4)))
}
z = 0
a <- -1 - 400*exp(-20*z - 4)/(1 + exp(-4*(5*z + 1))) + 400*exp(-40*z - 8)/(1 + exp(-4*(5*z + 1)))**2
a <- -a
kernel <- function (x) {
  exp(-(a/2) * ((x)**2))
}
const <- integrate(kernel, -2, 4)
q <- function (x, z = 0) {
  a <- -1 - 400*exp(-20*z - 4)/(1 + exp(-4*(5*z + 1))) + 400*exp(-40*z - 8)/(1 + exp(-4*(5*z + 1)))**2
  a <- -a
  sqrt(a / (2*pi)) * exp(-a * (((x - z)**2) / 2))
}

x <- seq(-2, 4, length.out = 200)
p1 <- p(x) / integrate(p, -2, 4)$value
q1 <- q(x) #kernel(x) / const$value
xyplot(p1 ~ x, type = c('l', 'g'), col = 'black', ylim = c(-0.1, 1.1),
       panel = function(x, y, ...) {
         panel.xyplot(x, y, ...)
         panel.xyplot(x, q1, col = 'black', type = 'l')
       })

curve(q(x))


dchisq2 <- function (x, df) {
  (x ** (df - 1) * exp(- (x**2) / 2)) / (2 ** (df/2 - 1) * gamma(df / 2))
}

k1 <- 2
x <- seq(0, 6, length = 200)
yp1 <- dchisq2(x, k1)
ya1 <- dnorm(x, sqrt(k1 - 1), sqrt(.5))

k2 <- 10
yp2 <- dchisq2(x, k2)
ya2 <- dnorm(x, sqrt(k2 - 1), sqrt(.5))

k3 <- 25
yp3 <- dchisq2(x, k3)
ya3 <- dnorm(x, sqrt(k3 - 1), sqrt(.5))


k4 <- 30
yp4 <- dchisq2(x, k4)
ya4 <- dnorm(x, sqrt(k4 - 1), sqrt(.5))

laplace_dat <- data.frame(
  X = rep(x, times = 4), DF = rep(c('DF = 2', 'DF = 10', 'DF = 25', 'DF = 30'), each = length(yp1)), 
  Posterior = c(yp1, yp2, yp3, yp4), Approximator = c(ya1, ya2, ya3, ya4)
)

library(lattice)
mytheme <- trellis.par.get()
mytheme$strip.background$col = 'gray'
mytheme$layout.heights$strip = 1.5
trellis.par.set(mytheme)
xyplot(Posterior + Approximator ~ X | factor(DF), data = laplace_dat, index.cond = list(c(3, 4, 2, 1)), 
       type = c('l', 'g'), lwd = 3, lty = c(1, 3), col = c('gray60', 'black'),
       key = list(lines = list(lty = c(1, 3), lwd = c(3, 3), col = c('gray60', 'black')),
                  text = list(labels = c('Posterior', 'Laplace')),
                  corner = c(.3, .8))
)

#### Bayesian Linear Regression
# Sequential Bayesian Learning
# Simulate the data

# Set the parameter
library(mvtnorm)
w0 <- -.3; w1 <- .5
x <- runif(20, -1, 1)
A <- 1 %>% rep(times = 20) %>% cbind(x)
B <- rbind(w0, w1)
f <- A %*% B 
y <- f + rnorm(20, sd = .2)
plot(x, y)



# FIRST: Prior
Imat <- 1 %>% diag(2, 2)
b <- 2
b1 <- (1 / b)**2 # Square this since in R, rnorm uses standard dev

mu <- 0 %>% rep(times = 2) %>% matrix(ncol = 1)
s <- b1 * Imat

x1 <- seq(-1, 1, length.out = 200)
x2 <- seq(-1, 1, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
z <- numeric()
for (i in 1:nrow(Grid)) {
  z[i] <- dmvnorm(x = c(Grid[i, 1], Grid[i, 2]), mu, s)
}
Grid$z <- z; z <- matrix(z, 200, 200, byrow = FALSE)

contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]), labels = TRUE, aspect = 'xy',
            panel = function(x, y, ...) {
              panel.contourplot(x, y, ...)
              panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
            }, colorkey = FALSE)

# Take 6 sample weights from prior
xseq <- seq(-1, 1, length.out = 100)
wprior <- 1 %>% rmvnorm(mean = mu, sigma = s)
y0 <- wprior[1, 1]; y1 <- wprior[1, 2]
f <- y0 + y1 * xseq
xyplot(f ~ xseq, type = 'l', col = 'gray60', xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1), aspect = 'yx',
       xlab = 'x', ylab = 'y',
       panel = function (x, y, ...) {
         panel.grid(h = -1, v = -1)
         panel.xyplot(x, y, ...)
         n <- 20
         for (i in 1:n) {
           wprior <- 1 %>% rmvnorm(mean = mu, sigma = s)
           y0 <- wprior[1, 1]; y1 <- wprior[1, 2]
           f <- y0 + y1 * xseq
           panel.xyplot(xseq, f, type = 'l', col = 'gray60')
         }
       })

# SECOND: Add single data point
# Likelihood for 1st data
x1 <- seq(-1, 1, length.out = 200) # weight space
x2 <- seq(-1, 1, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
alpha <- (1/.2)**2 # since the sd in Gaussian noise is .2
z <- numeric()

for (i in 1:nrow(Grid)) {
  z[i] <- dnorm(x = y[5,], sum(A[1,] * Grid[i,]), alpha)
}

Grid$z <- z; z <- matrix(z, 200, 200, byrow = FALSE)

library(plot3D)
contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]), labels = TRUE, aspect = 'xy',
            panel = function(x, y, ...) {
              panel.contourplot(x, y, ...)
              panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
            }, colorkey = FALSE)

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

library(plot3D)
contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]), labels = TRUE, aspect = 'xy',
            panel = function(x, y, ...) {
              panel.contourplot(x, y, ...)
              panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
            }, colorkey = FALSE)

# Likelihood for 20th data
x1 <- seq(-1, 1, length.out = 200) # weight space
x2 <- seq(-1, 1, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
alpha <- (1/.2)**2 # since the sd in Gaussian noise is .2
z <- numeric()

# Likelihood for 20th data
for (i in 1:nrow(Grid)) {
  z[i] <- dnorm(x = y[20,], sum(A[20,] * Grid[i,]), alpha)
}

Grid$z <- z; z <- matrix(z, 200, 200, byrow = FALSE)

# Likelihood 
library(plot3D)
contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]), labels = TRUE, aspect = 'xy',
            panel = function(x, y, ...) {
              panel.contourplot(x, y, ...)
              panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
            }, colorkey = FALSE)


# Posterior for 1st data
A1 <- A[1,] %>% matrix(ncol = 2)
s1 <- alpha * ((A1 %>% t %*% A1)) + b * Imat
mu1 <- (alpha * (s1) %>% solve) %*% (A1 %>% t) * y[1,]
wprior <- 1 %>% rmvnorm(mean = mu1, sigma = solve(s1))

x1 <- seq(-1, 1, length.out = 200)
x2 <- seq(-1, 1, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
z <- numeric()
for (i in 1:nrow(Grid)) {
  z[i] <- dmvnorm(x = c(Grid[i, 1], Grid[i, 2]), mu1, solve(s1))
}
Grid$z <- z; z <- matrix(z, 200, 200, byrow = FALSE)

library(plot3D)
contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]), labels = TRUE, aspect = 'xy',
            panel = function(x, y, ...) {
              panel.contourplot(x, y, ...)
              panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
            }, colorkey = FALSE)
persp3D(x1, x2, z, col = ramp.col(c('gray', 'gray10')), colkey = FALSE, 
        theta = 230, phi = 20, lighting = TRUE, ltheta = 160, lphi = 0,
        xlab = '', ylab = '')#, ticktype = 'detailed')

xyplot(y[1] ~ x[1], type = 'p', col = 'black', xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1), aspect = 'yx',
       xlab = 'x', ylab = 'y', cex = 1.2, pch = 19,
       panel = function (x, y, ...) {
         panel.grid(h = -1, v = -1)
         xseq <- seq(-1, 1, length.out = 100)
         n <- 20
         for (i in 1:n) {
           wprior <- 1 %>% rmvnorm(mean = mu1, sigma = solve(s1))
           y0 <- wprior[1, 1]; y1 <- wprior[1, 2]
           f <- y0 + y1 * xseq
           panel.xyplot(xseq, f, type = 'l', col = 'gray60')
         }
         panel.xyplot(x, y, ...)
       })


# Posterior for 5th data
A2 <- A[1:5,] %>% matrix(ncol = 2)
s2 <- alpha * ((A2 %>% t %*% A2)) + b * Imat
yy <- y[1:5,] %>% matrix
yy
mu2 <- (alpha * (s2) %>% solve) %*% (A2 %>% t) %*% yy
wprior <- 1 %>% rmvnorm(mean = mu2, sigma = solve(s2))

x1 <- seq(-1, 1, length.out = 200)
x2 <- seq(-1, 1, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
z <- numeric()
for (i in 1:nrow(Grid)) {
  z[i] <- dmvnorm(x = c(Grid[i, 1], Grid[i, 2]), mu2, solve(s2))
}
Grid$z <- z; z <- matrix(z, 200, 200, byrow = FALSE)

library(plot3D)
contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]), labels = TRUE, aspect = 'xy',
            panel = function(x, y, ...) {
              panel.contourplot(x, y, ...)
              panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
            }, colorkey = FALSE)
persp3D(x1, x2, z, col = ramp.col(c('gray', 'gray10')), colkey = FALSE, 
        theta = 230, phi = 20, lighting = TRUE, ltheta = 160, lphi = 0,
        xlab = '', ylab = '')#, ticktype = 'detailed')

xyplot(y[1:5] ~ x[1:5], type = 'p', xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1), aspect = 'yx',
       xlab = 'x', ylab = 'y', cex = 1.2, pch = 19, col = 'black',
       panel = function (x, y, ...) {
         panel.grid(h = -1, v = -1)
         xseq <- seq(-1, 1, length.out = 100)
         n <- 20
         for (i in 1:n) {
           wprior <- 1 %>% rmvnorm(mean = mu2, sigma = solve(s2))
           y0 <- wprior[1, 1]; y1 <- wprior[1, 2]
           f <- y0 + y1 * xseq
           panel.xyplot(xseq, f, type = 'l', col = 'gray60')
         }
         panel.xyplot(x, y, ...)
       })


# Posterior for 20th data
A2 <- A[1:20,] %>% matrix(ncol = 2)
s2 <- alpha * ((A2 %>% t %*% A2)) + b * Imat
yy <- y[1:20,] %>% matrix
yy
mu2 <- (alpha * (s2) %>% solve) %*% (A2 %>% t) %*% yy
wprior <- 1 %>% rmvnorm(mean = mu2, sigma = solve(s2))

x1 <- seq(-1, 1, length.out = 200)
x2 <- seq(-1, 1, length.out = 200)
Grid <- expand.grid(y1 = x1, y2 = x2)
z <- numeric()
for (i in 1:nrow(Grid)) {
  z[i] <- dmvnorm(x = c(Grid[i, 1], Grid[i, 2]), mu2, solve(s2))
}
Grid$z <- z; z <- matrix(z, 200, 200, byrow = FALSE)

library(plot3D)
contourplot(z ~ y1 * y2, Grid, cuts = 10,  col.regions = ramp.col(), region = TRUE, xlab = expression(w[0]), ylab = expression(w[1]), labels = TRUE, aspect = 'xy',
            panel = function(x, y, ...) {
              panel.contourplot(x, y, ...)
              panel.points(w0, w1, pch = 23, fill = 'white', col = 'white', cex = 1.2)
            }, colorkey = FALSE)
persp3D(x1, x2, z, col = ramp.col(c('gray', 'gray10')), colkey = FALSE, 
        theta = 230, phi = 20, lighting = TRUE, ltheta = 160, lphi = 0,
        xlab = '', ylab = '')#, ticktype = 'detailed')

xyplot(y[1:20] ~ x[1:20], type = 'p', xlim = c(-1.1, 1.1), ylim = c(-1.1, 1.1), aspect = 'yx',
       xlab = 'x', ylab = 'y', cex = 1.2, pch = 19, col = 'black',
       panel = function (x, y, ...) {
         panel.grid(h = -1, v = -1)
         xseq <- seq(-1, 1, length.out = 100)
         n <- 20
         for (i in 1:n) {
           wprior <- 1 %>% rmvnorm(mean = mu2, sigma = solve(s2))
           y0 <- wprior[1, 1]; y1 <- wprior[1, 2]
           f <- y0 + y1 * xseq
           panel.xyplot(xseq, f, type = 'l', col = 'gray60')
         }
         panel.xyplot(x, y, ...)
       })

### Metropolis - Hasting Algorithm
x <- seq(100, 200, .001)
y<-(sqrt(
  cos(x)
  )
    *cos(200*x)+sqrt(abs(x))-0.7)*(4-x*x)^0.01
y
