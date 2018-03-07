workspace();

using PyPlot
using StatsBase: autocor
using StatsBase: fit, Histogram
using Distributions
using StochMCMC


nsample = 80000;
xStep = 0.01;
m = 1; # dK
C = 3;
dt = 0.1;
nstep = 50;
V = 4;


srand(1234);

r = 80000;
xStep = 0.01;
ɛ = .1;
τ = 50;

# set up functions
U(x) = -2(x.^2) + x.^4;
dU(x) = -4x + 4(x.^3);
dU2(x) = -4x + 4(x.^3) + 2*rand(Normal());
K(p) = (p * p) / 2;
dK(p) = p;
H(x, p) = U(x) + K(p);


#gradUPerfect(x) = -4x + 4(x.^3);
"""
HMC without noise with M-H
"""
x1 = Array{Float64, 1}(); push!(x1, 0)

for n in 1:(r - 1)
  xNew = x1[n]
  p = rand(Normal())
  oldE = H(xNew, p)

  for τ_i in 1:τ
    p = p - (ɛ / 2) * dU(xNew)
    xNew = xNew + ɛ * dK(p)
    p = p - (ɛ / 2) * dU(xNew)
  end

  newE = H(xNew, p)
  dE = newE - oldE

  if dE < 0
    push!(x1, xNew)
  elseif rand(Uniform()) < exp(-dE)
    push!(x1, xNew)
  else
    push!(x1, x1[n])
  end

end

"""
HMC without noise with no M-H
"""
x2 = Array{Float64, 1}(); push!(x2, 0)

for n in 1:(r - 1)
  xNew = x2[n]
  p = rand(Normal())
  oldE = H(xNew, p)

  for τ_i in 1:τ
    p = p - (ɛ / 2) * dU(xNew)
    xNew = xNew + ɛ * dK(p)
    p = p - (ɛ / 2) * dU(xNew)
  end

  push!(x2, xNew)
end

"""
HMC with noise with M-H
"""
x3 = Array{Float64, 1}(); push!(x3, 0)

for n in 1:(r - 1)
  xNew = x3[n]
  p = rand(Normal())
  oldE = H(xNew, p)

  for τ_i in 1:τ
    p = p - (ɛ / 2) * dU2(xNew)
    xNew = xNew + ɛ * dK(p)
    p = p - (ɛ / 2) * dU2(xNew)
  end

  newE = H(xNew, p)
  dE = newE - oldE

  if dE < 0
    push!(x3, xNew)
  elseif rand(Uniform()) < exp(-dE)
    push!(x3, xNew)
  else
    push!(x3, x3[n])
  end

end

"""
HMC without noise with no M-H
"""
x4 = Array{Float64, 1}(); push!(x4, 0)

for n in 1:(r - 1)
  xNew = x4[n]
  p = rand(Normal())
  oldE = H(xNew, p)

  for τ_i in 1:τ
    p = p - (ɛ / 2) * dU2(xNew)
    xNew = xNew + ɛ * dK(p)
    p = p - (ɛ / 2) * dU2(xNew)
  end

  push!(x4, xNew)
end

"""
Stochastic Gradient HMC
"""
V = 4;
x5 = Array{Float64, 1}(); xNew = 0
p = rand(Normal()) * sqrt(1);
for n in 1:(r - 1)
  B = 0.5 * V * ɛ;
  D = sqrt(2 * (C - B) * ɛ);

  for i = 1:τ
    p = p - dU2(xNew) * ɛ  - p * C * ɛ  + rand(Normal()) * D;
    xNew = xNew + dK(p) * dt;
  end

  push!(x5, xNew)
end

"""
Plot the Target Distribution and HMCs
"""
xGrid = collect(-3:xStep:3);
y = exp(-U(xGrid));
y = y / sum(y) / xStep;
xhmc1, yhmc1 = hist(x1, xGrid);
yhmc1 = yhmc1 / sum(yhmc1) / xStep;

xhmc2, yhmc2 = hist(x2, xGrid);
yhmc2 = yhmc2 / sum(yhmc2) / xStep;

xhmc3, yhmc3 = hist(x3, xGrid);
yhmc3 = yhmc3 / sum(yhmc3) / xStep;

xhmc4, yhmc4 = hist(x4, xGrid);
yhmc4 = yhmc4 / sum(yhmc4) / xStep;

xhmc5, yhmc5 = hist(x5, xGrid);
yhmc5 = yhmc5 / sum(yhmc5) / xStep;
xyplot(xGrid, y, linestyle = "-", marker = "", save = false);
xyplot(xhmc1[2:end], yhmc1, linestyle = "-", marker = "", add = true, save = false);
xyplot(xhmc2[2:end], yhmc2, linestyle = "-", marker = "", add = true, save = false);
xyplot(xhmc3[2:end], yhmc3, linestyle = "-", marker = "", add = true, save = false);
xyplot(xhmc4[2:end], yhmc4, linestyle = "-", marker = "", add = true, save = false);
xyplot(xhmc5[2:end], yhmc5, linestyle = "-", marker = "", add = true);

"""
Plot the Autocorrelation
"""

barplot(autocor(x1))
barplot(autocor(x2))
barplot(autocor(x3))
barplot(autocor(x4))
barplot(autocor(x5))

histogram(x)
xs = collect(1.:size(x)[1])
x
1:size(x)[1]
%% set up functions
U = @(x) (-2* x.^2 + x.^4);
gradU = @(x) ( -4* x +  4*x.^3) +  randn(1) * 2;
gradUPerfect =  @(x) ( - 4*x +  4*x.^3 );
fgname = 'figure/func4';
hmccmp;

"""
COMPARISON OF HAMILTONIAN MONTE CARLOS
-------------------------------------

Setup the HMC functions:
"""
# Define Potential Energy Function
U(x, Σ) = x' * inv(Σ) * x

# Gradient of Potential Energy Function
dU(x, Σ) = inv(Σ) * x

# Define Kinetic Energy
K(p) = (p' * p) / 2

function hmc(U::Function, dU::Function, K::function; δ::Float64 = .3, n::Int64 = 10000, L::Int64 = 20)
  # Initial State
  x = zeros(2, n)
  x0 = [0, 6]
  x[:, 1] = x0

  # Sigma for dU
  Σ = [1 .8; .8 1]

  t = 1
  while t < n
    t = t + 1

    # Simulate Random Momentum
    p0 = rand(Normal(), 2)

    # Simulate Hamiltonian Dynamics
    # --------------------------------
    pStar = p0 - (δ / 2) * dU(x[:, t - 1], Σ)

    # Simulate
    xStar = x[:, t - 1] + δ * pStar;

    # Full Steps
    for jL = 1:L - 1

      # Momentum
      pStar = pStar - δ * dU(pStar, Σ)

      # Position
      xStar = xStar + δ * pStar

    end

    # Last Half Step
    pStar = pStar - (δ / 2) * dU(pStar, Σ)

    U0 = U(x[:, t - 1], Σ)
    UStar = U(xStar, Σ)

    K0 = K(p0)
    KStar = K(pStar)

    α = float(min(1, exp((U0 + K0) - (UStar + KStar))))

    u = rand(Uniform())

    if u < α[1]
      x[:, t] = xStar
    else
      x[:, t] = x[:, t - 1]
    end

  end
end

function hmc(U, gradU, m, dt, nstep, x, mhtest)
  p = randn( size(x) ) * sqrt( m );
  oldX = x;
  oldEnergy = p' * m * p / 2 + U(x);
  # do leapfrog
  for i = 1 : nstep
    p = p - gradU( x ) * dt / 2;
    x = x + p./m * dt;
    p = p - gradU( x ) * dt / 2;
  end

  p = -p;

  # M-H test
  if mhtest ~= 0
    newEnergy  = p' * m * p / 2 + U(x);

    if exp(oldEnergy- newEnergy) < rand(1)
        # reject
        x = oldX;
    end
  end
  newx = x;
end




"""
HMC without noise with Metropolis-Hasting
"""
samples = zeros(nsample, 1);
x = 0;
for i = 1:nsample
    x = hmc(U, gradUPerfect, m, dt, nstep, x, 1);
    samples[i] = x;
end
[yhmc, xhmc] = hist(samples, xGrid);
yhmc = yhmc / sum(yhmc) / xStep;
plot( xhmc, yhmc, 'c-v');

using Distributions

srand(12345);
x = rand(Normal(15, 2), 100);
beta = [3.4; .75];
err = rand(Normal(), 100);
X = [[1 for i in 1:size(x)[1]] x];
y = X * beta + err;

using GLM
using DataFrames
dat = DataFrame(X = x, Y = y)

fit(LinearModel, Y ~ X, dat)

xyplot(x, y)
beta
X
set.seed(12345)
x <- rnorm(100, 15, 2); beta <- rbind(3.4, .75)
err <- rnorm(100); X <- cbind(1, x)
y <- X %*% beta + err



"""
SGHMC FOR GAUSSIAN DISTRIBUTION
"""
function sghmc2d(gradU, eta, L, alpha, x, V)
  m = length(x)
  data = zeros(m, L)
  beta = V * eta * .5

  if beta > alpha
    error("too big eta")
  end

  sigma = sqrt(2 * eta * (alpha - beta))
  p = randn(m, 1) * sqrt(eta)
  momentum = 1 - alpha

  for t = 1:L
    p = p * momentum - gradU(x) * eta + randn(2, 1) * sigma
    x = x + p
    data[:, t] = x
  end

  return data
end



#This file produces Figure 3b, running trace of SGLD and SGHMC
# parameters

#clear all;
global covS;
global invS;

V = 1;
# covariance matrix
rho = 0.9;
covS = [[1 rho]; [rho 1]];
invS = inv( covS );
# intial
x = [0;0];

#this is highest value tried so far for SGLD that does not diverge
etaSGLD = 0.05;
etaSGHMC = 0.05;
alpha = 0.035;
# number of steps
L = 50000;

probUMap(X,Y) = exp( - 0.5 *( X .* X * invS(1,1) + 2 * X.*Y*invS(1,2) + Y.* Y *invS(2,2) )) / ( 2*pi*sqrt(abs(det (covS))));
funcU(x) = 0.5 * (x - 10)'*invS*(x - 10);
gradUTrue(x) = invS * x;
gradUNoise(x) = invS * (x - 50)  + randn(2,1);

[XX,YY] = meshgrid( linspace(-2,2), linspace(-2,2) );
ZZ = probUMap( XX, YY );
contour( XX, YY, ZZ );
hold on;
% set random seed
randn( 'seed',20 );

dsgld = sgld( gradUNoise, etaSGLD, L, x, V );
@time dsghmc = sghmc2d( gradUNoise, etaSGHMC, L, alpha, x, V );

h1=scatter( dsgld(1,:), dsgld(2,:), 'bx');
h2=scatter( dsghmc(1,:), dsghmc(2,:), 'ro' );

mapslices(mean, dsghmc, [2])
xyplot(dsghmc[1, :], dsghmc[2, :])
barplot(autocor(dsghmc[1, 1000:end]))
barplot(autocor(dsghmc[2, 1000:end]))

xlabel('x');
ylabel('y');
legend([h1 h2], {'SGLD', 'SGHMC'});
axis([-2.1 3 -2.1 3]);
len = 4;
set(gcf, 'PaperPosition', [0 0 len len/8.0*6.5] )
set(gcf, 'PaperSize', [len len/8.0*6.5] )
fgname = 'figure/sgldcmp-run';
saveas( gcf, fgname, 'fig');
saveas( gcf, fgname, 'pdf');

function [ data ] = sghmc( gradU, eta, L, alpha, x, V )
%% SGHMC using gradU, for L steps, starting at position x,
%% return data: array of positions
m = length(x);
data = zeros( m, L );
beta = V * eta * 0.5;

if beta > alpha
    error('too big eta');
end

sigma = sqrt( 2 * eta * (alpha-beta) );
p = randn( m, 1 ) * sqrt( eta );
momentum = 1 - alpha;

for t = 1 : L
    p = p * momentum - gradU( x ) * eta + randn(2,1)* sigma;
    x = x + p;
    data(:,t) = x;
end

end
