import numpy as np
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import json


# Binary search
def binary_search(data, target):
    left, right = 0, len(data) - 1
    while left <= right:
        mid = (left + right) // 2
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Interpolation search
def interpolation_search(data, target):
    low, high = 0, len(data) - 1
    while low <= high and target >= data[low] and target <= data[high]:
        if low == high:
            if data[low] == target:
                return low
            return -1
        pos = low + int(((high - low) * (target - data[low]) / (data[high] - data[low])))
        if data[pos] == target:
            return pos
        elif data[pos] < target:
            low = pos + 1
        else:
            high = pos - 1
    return -1

# Exponential search
def exponential_search(data, target):
    if len(data) == 0:
        return -1
    if data[0] == target:
        return 0
    index = 1
    while index < len(data) and data[index] <= target:
        index *= 2
    return binary_search_range(data, target, index // 2, min(index, len(data)))

def binary_search_range(data, target, left, right):
    while left < right:
        mid = left + (right - left) // 2
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            left = mid + 1
        else:
            right = mid
    return -1

# datset sizes loop
results = {}

for n in range(100, 2001, 100):
    #print(f"\ntesting with dataset size: {n}")
    
    # generate datasets
    np.random.seed(42)  # For consistent datasets
    uniform_data = np.linspace(10, 100, n)
    skewed_data = np.random.exponential(scale=20, size=n)
    non_uniform_data = np.concatenate([
        np.random.uniform(10, 30, n // 3),
        np.random.uniform(60, 80, n // 3),
        np.random.uniform(90, 100, n // 3)
    ])

    datasets = {"Uniform": uniform_data, "Skewed": skewed_data, "NonUniform": non_uniform_data}

    # target value
    target = 60

    # Inject the target value into datasets
    for data in datasets.values():
        if len(data) > 10: 
            data[50] = target
        data.sort()

    # search methods
    search_methods = {
        "Binary Search": binary_search,
        "Interpolation Search": interpolation_search,
        "Exponential Search": exponential_search,
    }

    # run search and compute execution times
    results[n] = {}
    iterations = 100_000
    for method_name, method in search_methods.items():
        results[n][method_name] = {}
        for dataset_name, dataset in datasets.items():
            total_time = 0
            for _ in range(iterations):
                start_time = time.time_ns()
                method(dataset, target)
                end_time = time.time_ns()
                total_time += (end_time - start_time)
            avg_time_ns = total_time // iterations
            results[n][method_name][dataset_name] = avg_time_ns
            #print(f"{method_name} - {dataset_name}: {avg_time_ns} ns")

# save results to plot
output_file = "performance.json"
with open(output_file, "w") as file:
    json.dump(results, file, indent=4)


# get results from json
input_file = "performance.json"
with open(input_file, "r") as file:
    results = json.load(file)

# plot
def save_individual_graphs(results, output_folder="individual_plots"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataset_sizes = sorted(results.keys(), key=int) 
    dataset_sizes_numeric = list(map(int, dataset_sizes))
    search_methods = list(next(iter(results.values())).keys())
    dataset_types = list(next(iter(next(iter(results.values())).values())).keys())
    
    # generate data
    for dataset_type in dataset_types:
        for method in search_methods:
            plt.figure(figsize=(10, 5))

            # plot actual runtimes
            avg_times = [
                results[size][method][dataset_type] 
                for size in dataset_sizes
            ]
            plt.plot(dataset_sizes_numeric, avg_times, label="Actual Runtime", marker='o')

            # calc asympton runtimes for comparison
            if method == "Binary Search":
                asymp_runtime = [np.log2(n) for n in dataset_sizes_numeric]
            elif method == "Interpolation Search":
                if dataset_type == "Uniform":
                    asymp_runtime = [np.log2(np.log2(n)) for n in dataset_sizes_numeric]
                else:
                    asymp_runtime = [n for n in dataset_sizes_numeric]
            elif method == "Exponential Search":
                asymp_runtime = [np.log2(n) for n in dataset_sizes_numeric]

            # asymp values for comparison
            max_actual = max(avg_times)
            asymp_runtime_normalized = [
                max_actual * val / max(asymp_runtime) for val in asymp_runtime
            ]

            # plot runtimes
            plt.plot(dataset_sizes_numeric, asymp_runtime_normalized, '--', label="asymp Runtime", color='r')

            plt.title(f"{method} Performance on {dataset_type} Dataset")
            plt.xlabel("Dataset Size")
            plt.ylabel("Average Execution Time (ns)")
            plt.legend()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)

            # save plots
            output_path = os.path.join(output_folder, f"{dataset_type}_{method.replace(' ', '_')}.jpg")
            plt.savefig(output_path, format="jpg", dpi=300)
            plt.close() 


save_individual_graphs(results)
print("Done")
