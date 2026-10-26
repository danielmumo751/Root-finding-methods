import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Real-world problem: Chemical Reaction Conversion Rate Model
# f(X) = 0 represents the target fractional conversion X in a reactor.
# Equation: X = 1 - exp(-k * tau * (1 - X)) -> rewritten as f(X) = 0
def conversion_model(X):
    k = 0.5   # reaction rate constant
    tau = 4.0 # residence time
    return X - (1 - np.exp(-k * tau * (1 - X)))

# Bisection: bracket conversion between 0 and 1 (0% to 100% conversion)
solution = root_scalar(
    conversion_model,
    bracket=[0.0, 0.99],
    method="bisect",
    xtol=1e-8
)

# Fixed-point form: X_new = g(X)
def g(X):
    k = 0.5
    tau = 4.0
    return 1 - np.exp(-k * tau * (1 - X))

X_val = 0.5
history = [X_val]

for _ in range(100):
    X_new = g(X_val)
    history.append(X_new)
    if abs(X_new - X_val) < 1e-8:
        break
    X_val = X_new

print(f"Bisection equilibrium conversion: {solution.root:.6f}")
print(f"Fixed-point equilibrium conversion: {history[-1]:.6f}")
print("Fixed-point iterations:", len(history) - 1)

# Plotting the conversion function
conversions = np.linspace(0.0, 0.99, 300)
plt.plot(conversions, conversion_model(conversions))
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.axvline(solution.root, color='red', linestyle='--', label=f'Root: {solution.root:.4f}')
plt.xlabel("Fractional Conversion (X)")
plt.ylabel("Model Residual f(X)")
plt.title("Chemical Reactor Equilibrium Conversion")
plt.legend()
plt.grid(True)
plt.show()