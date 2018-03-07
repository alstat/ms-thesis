import numpy as np
import scipy.stats as st
import scipy as sp
import matplotlib.pylab as plt
plt.style.use('ggplot')

# Monte-Carlo Integration
# -----------------------------
# Suppose we want to integrate the area of the normal curve
# between -1.96 and 1.96. And assuming it is difficult to
# carry out the integration. We can use Monte-Carlo Integration
# as an approximator which uses sampling. And as the sample goes
# large it can better approximate the distribution.
draws = np.arange(1000, 100000, 100)
area = np.zeros(draws.shape)b

for i in np.arange(draws.shape[0]):
    sample = np.random.normal(size = draws[i])
    area[i] = np.sum((sample > -1.96) & (sample < 1.96)) / sample.shape[0]

plt.plot(draws, area)

# Accept-Reject Sampling
# -----------------------------
# Target: Cauchy Distribution
def cauchy(x, x_0 = 0, gamma = 1):
    """Univariate Cauchy Distribution

    x input
    x_0 location parameter
    gamma some parameter

    """
    out = 1 / (np.pi * gamma * ((1 + ((x - x_0) / gamma)**2)))
    return out

# Proposal: Standard Normal Distribution
# Define the envelope function
def e(x, k = 2, loc = 0, scale = 1):
    return k * st.norm.pdf(x, loc, scale)

# Accept-Reject Algorithm
def ar_algorithm(n = 10000):
    theta = np.zeros(n)

    for i in np.arange(n - 1):
        theta[i] = np.random.normal(size = 1)
        u = np.random.uniform(size = 1)

        if u < (cauchy(theta[i]) / e(theta[i])):
            theta[i + 1] = theta[i]
        else:
            theta[i + 1] = np.nan

    return theta[ ~ np.isnan(theta)]

out = ar_algorithm()
plt.hist(out, normed = True, bins = 30)

x = np.linspace(-4, 4, 100)
y = cauchy(x)

plt.plot(x, y, color = 'black', lw = 2)
plt.xlim([-4.5, 4.5])
plt.ylim(-.03)
plt.xlabel('\nx')
plt.ylabel('Histogram of Accept-Reject Samples\n')

# Metropolis Algorithm
# ---------------------------------
# The following code approximates p(x), the standard normal distribution,
# using q(x), the proposal distribution with standard deviation of .05.

n = 250000
x = np.zeros(n)

# Initialize
x[0] = .5
for i in np.arange(n - 1):
    x_q = np.random.normal(x[i], .05) # Proposal distribution
    u_o = np.random.uniform(0, 1)
    if u_o < np.min([1, st.norm.pdf(x_q) / st.norm.pdf(x[i])]): # True distribution
        x[i + 1] = x_q
    else:
        x[i + 1] = x[i]

x_val = np.linspace(-4, 4, 100)
y_val = st.norm.pdf(x_val)
plt.hist(x[20000:], normed = True, bins = 50)
plt.plot(x_val, y_val)
plt.ylim([-.01, .45])
plt.xlim([-4.5, 4.5])

# Metropolis-Hasting Algorithm
# ---------------------------------
# Target Distribution: Cauchy
np.random.seed(1234)
reps = 100000

# Define Cauchy
def cauchy(x, x_0 = 0, gamma = 1):
    out = 1 / (np.pi * gamma * ((1 + ((x - x_0) / gamma)**2)))
    return out

chain = np.zeros((reps))
chain[0] = 30

for i in np.arange(reps - 1):
    proposal = chain[i] + np.random.uniform(-1, 1, 1)
    accept = np.random.uniform(size = 1) < (cauchy(proposal) / cauchy(chain[i]))
    if accept == True:
        chain[i + 1] = proposal
    else:
        chain[i + 1] = chain[i]

# Multivariate Cauchy
def multcauchy (x, x_0 = [0,0], gamma=1):
    out = (1 / (2 * np.pi)) * (gamma / ((x[0]-x_0[0])**2+(x[1]-x_0[1])**2 + gamma**2)**1.5)
    return out

chain = np.zeros((reps, 2))
chain[0, :] = [-100, 100]
for i in np.arange(reps - 1):
    proposal = chain[i, :] + np.random.uniform(-5, 5, 2)
    accept = np.random.uniform(size = 1) < (multcauchy(proposal) / multcauchy(chain[i]))
    if accept == True:
        chain[i + 1] = proposal
    else:
        chain[i + 1] = chain[i]
plt.figure(figsize = (6, 6))
plt.plot(chain[:,0], chain[:,1])
plt.savefig('A1.pdf')

# Gibbs Sampler
# ---------------------------------
np.random.seed(12345)
reps = 20000

def con_norm(x_2, loc_1 = 10, loc_2 = -10, scale_1 = 1.5, scale_2 = 1.35, rho = .25):
    out = np.random.normal(loc = loc_1 + (scale_1/scale_2) * rho * (x_2 - loc_2), scale = np.sqrt((1 - (rho**2)) * (scale_1**2)), size = 1)
    return out

chain_1 = np.zeros(reps)
chain_2 = np.zeros(reps)

for i in np.arange(reps - 1):
    chain_1[i + 1] = con_norm(chain_2[i])
    chain_2[i + 1] = con_norm(chain_1[i + 1], -10, 10, 1.35, 1.5)

plt.plot(chain_1[1000:20000], chain_2[1000:20000], 'o')

# Hamiltonian Monte Carlo Sampler
# ---------------------------------

# Step Size
delta = .3
nSamples = 1000
L = 20

# Define Potential Energy Function
def U(x, S):
    x.T.dot(np.linalg.inv(S)).dot(x)

# Define Gradient of Potential Energy
def dU(x, S):
    x.T.dot(np.linalg.inv(S))

# Define Kinetic Energy Function
def K(p):
    np.sum(p.T.dot(p)) / 2

# Initial State
x = np.zeros((2, nSamples))
x0 = [0, 6]
x[:, 0] = x0

mean = np.array([0, 0])
var = np.array([[1, 0], [0, 1]])
S = np.array([[1, .8], [.8, 1]])
S
t = 0
while t < nSamples:
    t = t + 1

    # Sample Random Momentum
    p0 = np.random.multivariate_normal(mean, var)

    # Simulate Hamiltonian Dynamics
    # -----------------------------

    # First 1/2 Step of Momentum
    pStar = p0 - (delta/2) * dU(x[:, t - 1], S)

    # First Full Step for Position/Sample
    xStar = x[:, t - 1] + delta * pStar

    # Full Steps
    for jL in np.arange(L - 1):

        # Momentum
        pStar = pStar - delta*dU(xStar)

        # Position/Sample
        xStar = xStar + delta*pStar

    # Last Half Step of Momentum
    pStar = pStar - delta/2 * dU(xStar)

    # Could Negate Momentum Here to Leave
    # The proposal distribution Symmetric.
    # However we throw this away for next sample,
    # so it doesn't matter

    # Evaluate energies at
    # Start and end of trajectory
    U0 = U(x[:, t - 1])
    UStar = U(xStar)

    K0 = K(p0)
    KStar = K(pStar)

    # Acceptance/Rejection Criterion
    alpha = np.min(1, np.exp((U0 + K0) - (UStar + KStar)))

    u = np.random.uniform(size = 1)
    if u < alpha:
        x[:, t] = xStar
    else:
        x[:, t] = x[:, t-1]



# Benchmark

def fib(n):
    if n<2:
        return n
    return fib(n-1)+fib(n-2)

def print_perf(name, time):
    print("python," + name + "," + str(time*1000))

assert fib(20) == 6765
tmin = float('inf')
for i in range(5):
    t = time.time()
    f = fib(20)
    t = time.time()-t
    if t < tmin: tmin = t
print_perf("fib", tmin)
