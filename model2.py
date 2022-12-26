import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint

# Total population, N.
N = 120000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0

# Observed data for the number of infected individuals over time
t_data = np.array(list(range(1, 346)))
I_data = np.array([1, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 3, 1, 0, 1, 1, 3, 0, 1, 2, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 4, 0, 3, 0, 2, 1, 3, 2, 0, 0, 0, 0, 0, 1, 1, 3, 2, 3, 1, 1, 1, 2, 2, 3, 3, 0, 0, 4, 0, 1, 3, 1, 3, 0, 5, 1, 2, 1, 1, 1, 0, 2, 1, 2, 3, 1, 0, 1, 0, 1, 4, 0, 1, 0, 2, 2, 1, 1, 2, 1, 0, 0, 1, 3, 2, 0, 1, 2, 0, 1, 1, 1, 0, 1, 2, 1, 0, 2, 3, 2, 0, 1, 1, 0, 0, 1, 3, 2, 0, 0, 3, 3, 4, 4, 2, 0, 0, 2, 1, 3, 2, 0, 1, 0, 0, 0, 1, 3, 1, 2, 2, 1, 2, 3, 1, 3, 5, 1, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1, 3, 6, 3, 1, 2, 1, 1, 4, 0, 3, 0, 1, 0, 0, 0, 2, 1, 1, 3, 4, 2, 3, 2, 3, 0, 7, 2, 2, 0, 3, 1, 0, 3, 2, 4, 7, 7, 4, 1, 1, 4, 4, 2, 4, 6, 0, 8, 9, 2, 7, 0, 5, 2, 6, 0, 0, 0, 4, 4, 0, 0, 1, 3, 1, 0, 5, 2, 2, 2, 3, 4, 4, 3, 0, 1, 2, 0, 4, 5, 5, 1, 6, 3, 2, 3, 1, 1, 1, 4, 0, 5, 2, 2, 5, 2, 8, 5, 2, 10, 2, 5, 1, 1, 3, 3, 5, 2, 4, 0, 2, 7, 3, 4, 5, 1, 0, 12, 6, 4, 3, 4, 7, 3, 7, 3, 3, 5, 7, 5, 0, 3, 1, 6, 5, 5, 3, 2, 5, 6, 7, 8, 5, 2, 4, 12, 13, 16, 19, 15, 25, 10, 48, 36, 61, 54, 38, 49, 15, 62, 49, 29, 47, 39, 32, 12, 43, 28, 25, 27, 23, 25, 5, 16, 11, 20, 8, 8, 11, 3, 16, 4, 10, 5, 1])

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Fit the model to the data
def fit_odeint(x, beta, gamma):
    return odeint(deriv, y0, x, args=(N, beta, gamma))[:,1]

params, params_covariance = curve_fit(fit_odeint, t_data, I_data)

print(params)
