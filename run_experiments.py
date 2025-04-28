#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SimplifiedMRF experiments with different labeled data percentages
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Experiment configurations
LABELED_PERCENTAGES = [10, 30, 75, 100]
BASE_EXPERIMENT_NAME = "EnhancedSimplifiedMRF"  # Updated name to reflect enhanced model
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.001
LATENT_DIM = 512  # Matches the enhanced CVAE latent dimension
SEED = 42

# Make sure output directory exists
os.makedirs("./output", exist_ok=True)

# Create timestamp for this experiment run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_run_name = f"{BASE_EXPERIMENT_NAME}_run_{timestamp}"
os.makedirs(f"./output/{experiment_run_name}", exist_ok=True)

# Results storage
results = []

# Run experiments
for percentage in LABELED_PERCENTAGES:
    experiment_name = f"{BASE_EXPERIMENT_NAME}_{percentage}pct_labeled"
    output_folder = f"./output/{experiment_name}"
    
    print(f"\n\n===== Running experiment with {percentage}% labeled data =====\n")
    
    # Build command
    cmd = [
        "python", "run_simplified_mrf.py",
        "-i", "./input/",
        "-o", output_folder,
        "-c", "./output/Enhanced-CVAE/model_best.pth",
        "-b", str(BATCH_SIZE),
        "-e", str(EPOCHS),
        "-lr", str(LEARNING_RATE),
        "-ld", str(LATENT_DIM), 
        "-lp", str(percentage),
        "-s", str(SEED),
        "-m", "train"
    ]
    
    # Run command
    subprocess.run(cmd)
    
    # Run validation after training
    val_cmd = [
        "python", "run_simplified_mrf.py",
        "-i", "./input/",
        "-o", output_folder,
        "-c", "./output/Enhanced-CVAE/model_best.pth",
        "-m", "validate",
        "-cp", f"{output_folder}/model_best.pth",
        "-lp", str(percentage),
        "-s", str(SEED)
    ]
    
    subprocess.run(val_cmd)
    
    # Read results from result.txt
    try:
        with open(f"{output_folder}/result.txt", "r") as f:
            lines = f.readlines()
            experiment_results = {
                "percentage": percentage,
                "accuracy": None,
                "mean_iou": None,
                "f1_score": None
            }
            
            for line in lines:
                if "Accuracy ->" in line:
                    experiment_results["accuracy"] = float(line.split("->")[1].strip().rstrip("%"))
                elif "Mean IoU ->" in line:
                    experiment_results["mean_iou"] = float(line.split("->")[1].strip().rstrip("%"))
                elif "F1 Score ->" in line:
                    experiment_results["f1_score"] = float(line.split("->")[1].strip().rstrip("%"))
            
            results.append(experiment_results)
            print(f"Recorded results for {percentage}% labeled data:")
            print(f"  Accuracy: {experiment_results['accuracy']}%")
            print(f"  Mean IoU: {experiment_results['mean_iou']}%")
            print(f"  F1 Score: {experiment_results['f1_score']}%")
    except Exception as e:
        print(f"Error reading results: {e}")

# Plot results
if results:
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(results)
    
    # Create figure with 3 subplots (one for each metric)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot accuracy
    axs[0].plot(df["percentage"], df["accuracy"], 'bo-', linewidth=2)
    axs[0].set_xlabel("Percentage of Labeled Data (%)")
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].set_title("SimplifiedMRF: Accuracy vs. Labeled Data")
    axs[0].grid(True)
    
    # Plot mean IoU
    axs[1].plot(df["percentage"], df["mean_iou"], 'go-', linewidth=2)
    axs[1].set_xlabel("Percentage of Labeled Data (%)")
    axs[1].set_ylabel("Mean IoU (%)")
    axs[1].set_title("SimplifiedMRF: Mean IoU vs. Labeled Data")
    axs[1].grid(True)
    
    # Plot F1 score
    axs[2].plot(df["percentage"], df["f1_score"], 'ro-', linewidth=2)
    axs[2].set_xlabel("Percentage of Labeled Data (%)")
    axs[2].set_ylabel("F1 Score (%)")
    axs[2].set_title("SimplifiedMRF: F1 Score vs. Labeled Data")
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"./output/{experiment_run_name}/labeled_data_experiment_results.png")
    
    # Save results to CSV for future analysis
    df.to_csv(f"./output/{experiment_run_name}/experiment_results.csv", index=False)
    
    # Generate a markdown report
    with open(f"./output/{experiment_run_name}/experiment_report.md", "w") as f:
        f.write(f"# SimplifiedMRF Labeled Data Experiment Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Experiment Configuration\n\n")
        f.write(f"- Model: SimplifiedMRF with pre-trained CVAE features\n")
        f.write(f"- Labeled Percentages: {LABELED_PERCENTAGES}\n")
        f.write(f"- Epochs: {EPOCHS}\n")
        f.write(f"- Batch Size: {BATCH_SIZE}\n")
        f.write(f"- Learning Rate: {LEARNING_RATE}\n")
        f.write(f"- Latent Dimension: {LATENT_DIM}\n")
        f.write(f"- Random Seed: {SEED}\n\n")
        
        f.write(f"## Results Summary\n\n")
        f.write("| Labeled Data (%) | Accuracy (%) | Mean IoU (%) | F1 Score (%) |\n")
        f.write("|-----------------|--------------|--------------|-------------|\n")
        
        for _, row in df.iterrows():
            f.write(f"| {row['percentage']:15d} | {row['accuracy']:12.2f} | {row['mean_iou']:12.2f} | {row['f1_score']:13.2f} |\n")
        
        f.write("\n\n")
        f.write("## Analysis\n\n")
        f.write("The experiment tested our SimplifiedMRF model's performance with different amounts of labeled training data. ")
        f.write("Our hypothesis was that the model can achieve good performance even with limited labeled data due to the ")
        f.write("use of pre-trained CVAE features that were learned in an unsupervised manner.\n\n")
        
        # Add baseline comparison section
        f.write("## Comparison to Baseline\n\n")
        f.write("Our target was to beat the baseline model's 85% accuracy using only 10% of labeled data. ")
        
        # Check if we achieved the target
        min_acc = df[df["percentage"] == 10]["accuracy"].values
        if len(min_acc) > 0 and min_acc[0] >= 85:
            f.write(f"**Target achieved!** Our model reached {min_acc[0]:.2f}% accuracy with only 10% labeled data.\n\n")
        else:
            f.write("The target was not achieved in this experiment run. Further improvements to the model or training process may be needed.\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("The experiment demonstrates how our two-stage approach (unsupervised CVAE pre-training followed by ")
        f.write("supervised MRF segmentation) can leverage unlabeled data to reduce the need for extensive manual labeling.\n")
    
    print(f"\nResults saved to ./output/{experiment_run_name}/experiment_results.csv")
    print(f"Report generated at ./output/{experiment_run_name}/experiment_report.md")
else:
    print("No results collected! Check for errors in the experiment runs.")