import numpy as np
import matplotlib.pyplot as plt

one_core_time = 242.927
core_counts = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16])
# # static
# output_path = "speedup_with_fit_static.png"
# speed_up = np.array([1, 
#             one_core_time/188.256, 
#             one_core_time/92.350, 
#             one_core_time/62.758, 
#             one_core_time/59.107, 
#             one_core_time/55.699, 
#             one_core_time/68.479, 
#             one_core_time/49.107, 
#             one_core_time/51.145])

# dynamic
output_path = "speedup_with_fit_dynamic.png"
speed_up = np.array([
    1,
    one_core_time/125.456,
    one_core_time/94.792,
    one_core_time/66.806,
    one_core_time/52.864,
    one_core_time/48.079,
    one_core_time/49.018,
    one_core_time/40.698,
    one_core_time/40.074
])

p = core_counts
S = speed_up

x = 1.0 / p
y = 1.0 / S

A = np.vstack([np.ones_like(x), x]).T
b, m = np.linalg.lstsq(A, y, rcond=None)[0]


B = b
F = 1-B

print(f"Fitted serial fraction f = {B:.4f}")
print(f"parallel fraction P = {F:.4f}")
print(f"Theoretical max speed-up S_inf = 1/1-F = {1/(1-F):.2f}")


p_fit = np.linspace(1, 16, 200)
S_fit = 1.0 / ( B + (1-B)/p_fit )


plt.figure(figsize=(8,5))
plt.plot(core_counts, speed_up, 'o-', label='Measured')
plt.plot(p_fit, S_fit, '--', lw=2, label=f'p={F:.3f}')
plt.title("Speed-Up vs Number of Cores")
plt.xlabel("Number of Cores (p)")
plt.ylabel("Speed-Up S(p)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(output_path)
plt.show()
