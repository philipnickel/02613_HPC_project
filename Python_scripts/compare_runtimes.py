import numpy as np
import matplotlib.pyplot as plt

core_counts = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16])
static_runtimes = [208.474, 
                   188.256, 
                   92.350, 
                   62.758, 
                   59.107, 
                   55.699, 
                   68.479, 
                   49.107, 
                   51.145]


dynamic_runtimes = [242.927, 
                    125.456, 
                    94.792, 
                    66.806, 
                    52.864, 
                    48.079, 
                    49.018, 
                    40.698, 
                    40.074 ]


plt.figure(figsize=(8, 5))
plt.plot(core_counts, static_runtimes, marker='o', label='Static Scheduling')
plt.plot(core_counts, dynamic_runtimes, marker='s', linestyle='--', label='Dynamic Scheduling')
plt.title('Runtime Comparison: Static vs Dynamic')
plt.xlabel('Number of Cores (p)')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compare_runtimes.png")
plt.show()
