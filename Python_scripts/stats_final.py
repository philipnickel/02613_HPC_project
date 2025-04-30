import pandas as pd
import matplotlib.pyplot as plt

# --- load data ---
data = pd.read_csv('data_final.csv')
# --- compute global stats ---
average_mean_temp = data['mean_temp'].mean()
average_std_temp  = data['std_temp'].mean()


count_above_18 = (data['pct_above_18'] >= 50).sum()
count_below_15 = (data['pct_below_15'] >= 50).sum()

output_path = "average_mean_temp.png"
plt.figure(figsize=(8, 4))
plt.hist(data['mean_temp'], bins=20, edgecolor='black')
plt.title('Distribution of Mean Temperatures')
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Buildings')
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f'Average of mean temperatures: {average_mean_temp:.2f} °C')
print(f'Average of temperature std. dev.: {average_std_temp:.2f} °C')
print(f'Buildings with ≥50% area >18 celsius: {count_above_18}')
print(f'Buildings with ≥50% area <15 celsius: {count_below_15}')
