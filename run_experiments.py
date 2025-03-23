# -*- coding: utf-8 -*-
"""
Run experiments with different labeled data percentages
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Experiment configurations
LABELED_PERCENTAGES = [10, 25, 50, 75, 100]
BASE_EXPERIMENT_NAME = "hierarchical_pgm"
EPOCHS = 30
GT_TYPE = "conncomp"
ERO_DISK = 8

# Make sure output directory exists
os.makedirs("./output", exist_ok=True)

# Run experiments
results = []
for percentage in LABELED_PERCENTAGES:
    experiment_name = f"{BASE_EXPERIMENT_NAME}_{percentage}pct_labeled"
    print(f"\n\n===== Running experiment with {percentage}% labeled data =====\n")
    
    # Build command
    cmd = [
        "python", "main_hierarchical_pgm.py",
        "-r",  # Retrain
        "-g", GT_TYPE,
        "-d", str(ERO_DISK),
        "-e", str(EPOCHS),
        "-exp", experiment_name,
        "-lp", str(percentage)
    ]
    
    # Run command
    subprocess.run(cmd)
    
    # Read accuracy from results file
    try:
        with open(f"./output/{experiment_name}/result.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "overall accuracy in r = 0" in line:
                    accuracy = float(line.split("->")[1].strip())
                    results.append((percentage, accuracy))
                    print(f"Recorded accuracy: {accuracy}%")
                    break
    except Exception as e:
        print(f"Error reading results: {e}")

# Plot results
if results:
    percentages, accuracies = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, accuracies, 'o-', linewidth=2)
    plt.xlabel("Percentage of Labeled Data")
    plt.ylabel("Overall Accuracy (%)")
    plt.title("Semi-Supervised Hierarchical PGM Performance")
    plt.grid(True)
    plt.savefig("./output/labeled_data_experiment_results.png")
    plt.show()
    
    # Save results to CSV
    with open("./output/experiment_results.csv", "w") as f:
        f.write("labeled_percentage,accuracy\n")
        for p, a in results:
            f.write(f"{p},{a}\n")
    
    print("Results saved to ./output/experiment_results.csv")