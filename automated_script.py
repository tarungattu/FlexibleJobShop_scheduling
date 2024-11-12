# Define the input list
input_list = [1.86, 8.54, 3.14, 5.09, 3.71, 6.83, 4.81, 2.46, 1.37, 11.24, 2.5, 6.7]

# Get the integer rank series with ranks starting from 0
rank_series = sorted(range(len(input_list)), key=lambda i: input_list[i])
integer_series = [rank_series.index(i) for i in range(len(input_list))]

# Apply the formula num % 3 + 1 on each number in the integer series to get the second series
mod_series = [num % 3 for num in integer_series]

# Print both series
print("Integer series based on rank (starting from 0):", integer_series)
print("Resulting series using formula num % 3 + 1:", mod_series)
