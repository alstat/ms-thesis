workspace();

using PyPlot

nsample = 80000;
xStep = 0.01;
m = 1;
C = 3;
dt = 0.1;
nstep = 50;
V = 4;

srand(1234);

# set up functions
U(x) = -2(x.^2) + x.^4;
gradU(x) = -4x + 4(x.^3) + randn(1)*2;
gradUPerfect(x) = -4x + 4(x.^3);

"""
COMPARISON OF HAMILTONIAN MONTE CARLO
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
Plot the Target Distribution
"""
xGrid = collect(-3:xStep:3);
y = exp(-U(xGrid));
y = y / sum(y) / xStep;
xyplot(:(y ~ xGrid), linestyle = "-", marker = "");

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
