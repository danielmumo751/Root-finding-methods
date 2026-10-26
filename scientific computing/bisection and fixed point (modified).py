import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Modified Real-World Case: Electronic Component Heat Sink
# f(T) = Internal power generation minus convective heat dissipation = 0
# Let f(T) = 40 * exp(-T/30) - (T - 20) = 0
def component_heat_balance(T):
    return 40 * np.exp(-T / 30) - (T - 20)

# Bisection: bracket a sign change
solution = root_scalar(
    component_heat_balance,
    bracket=[10, 50],
    method="bisect",
    xtol=1e-10
)

# Fixed-point form: T = 20 + 40 * exp(-T/30)
def g(T):
    return 20 + 40 * np.exp(-T / 30)

T = 25.0
history = [T]

for _ in range(100):
    T_new = g(T)
    history.append(T_new)
    if abs(T_new - T) < 1e-10:
        break
    T = T_new

print(f"Modified Bisection equilibrium temperature: {solution.root:.6f} C")
print(f"Modified Fixed-point equilibrium temperature: {history[-1]:.6f} C")
print("Fixed-point iterations:", len(history) - 1)

temps = np.linspace(10, 50, 300)
plt.plot(temps, component_heat_balance(temps))
plt.axhline(0, color='red', linewidth=0.8, linestyle='--')
plt.axvline(solution.root, color='green', linestyle='--')
plt.xlabel("Component Temperature (C)")
plt.ylabel("Net Heat Function")
plt.title("Electronic Component Thermal Equilibrium")
plt.grid(True)
plt.show()