import matplotlib.pyplot as plt


core_counts = [1,2,4,6,8,10,12,14,16]
# #static
# output_path = "speedup_static.png"
# one_core_time = 208.474
# runtimes = [0, 
#             one_core_time/188.256, 
#             one_core_time/92.350, 
#             one_core_time/62.758, 
#             one_core_time/59.107, 
#             one_core_time/55.699, 
#             one_core_time/68.479, 
#             one_core_time/49.107, 
#             one_core_time/51.145]


#dynamic
output_path = "speedup_dynamic.png"
one_core_time = 242.927
runtimes = [0, 
            one_core_time/125.456, 
            one_core_time/94.792, 
            one_core_time/66.806, 
            one_core_time/52.864, 
            one_core_time/48.079, 
            one_core_time/49.018, 
            one_core_time/40.698, 
            one_core_time/40.074 ]

plt.figure(figsize=(8, 5))
plt.plot(core_counts, runtimes, marker='o')
plt.title("Speed-Up vs Number of Cores ")
plt.xlabel("Number of Cores (p)")
plt.ylabel("Speed Up (S(p)) ")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path)
plt.close()
print(f"Plot saved to {output_path}")