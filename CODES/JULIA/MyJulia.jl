# This is my favorite/ language so far
# Normal Density
f(x, mu, sigma) = 1 / (sqrt(2pi*sigma^2)) * exp(-(x - mu)^2 / (2sigma^2))

# Hamiltonian Monte Carlo
# ----------------------------

# Step Size
delta = .3
nSamples = 1000
L = 20

x = [1; 2]
s = eye(2)

println(transpose(x) * inv(s) * x)


# Define Potential Energy Function
U(x, S) = transpose(x) * inv(S) * x
U(x, s)


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
