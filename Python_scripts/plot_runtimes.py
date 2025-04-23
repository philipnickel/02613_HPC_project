import matplotlib.pyplot as plt

one_core_time = 242.927
core_counts = [1,2,4,6,8,10,12]
runtimes = [0, one_core_time/125.456, one_core_time/94.792, one_core_time/66.806, one_core_time/52.864, one_core_time/48.079, one_core_time/49.018]

print(one_core_time/125.456)
output_path = "speed-up-times.pdf"
plt.figure(figsize=(8, 5))
plt.plot(core_counts, runtimes, marker='o')
plt.title("Speed-Up vs Number of Cores ")
plt.xlabel("Number of Cores (p)")
plt.ylabel("Speed Up (S(p)) ")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path)  # 🔥 Save as PDF
plt.close()
print(f"Plot saved to {output_path}")
