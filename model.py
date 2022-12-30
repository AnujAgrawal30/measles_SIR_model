import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 120
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
# beta, gamma = 1.0027486, 0.99179328
beta, gamma = 0.1995502, 0.0760068
# A grid of time points (in days)
t = np.linspace(0, 70, 70)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T
data = np.array([1, 0, 12, 6, 4, 3, 4, 7, 3, 7, 3, 3, 5, 7, 5, 0, 3, 1, 6, 5, 5, 3, 2, 5, 6, 7, 8, 5, 2, 4, 12, 13, 16, 19, 15, 25, 10, 48, 36, 61, 54, 38, 49, 15, 62, 49, 29, 47, 39, 32, 12, 43, 28, 25, 27, 23, 25, 5, 16, 11, 20, 8, 8, 11, 3, 16, 4, 10, 5, 1])

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.plot(t, data, 'o', alpha=0.5, lw=2, label='Real data')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number')
ax.set_ylim(0,200)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
