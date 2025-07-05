

import os
import random
import subprocess
import re
from utils.ablation_tracker import log_ablation

# --- Configuration ---
NUM_TRIALS = 20
EPOCHS_PER_TRIAL = 15

# --- Hyperparameter Search Space ---
search_space = {
    'contrastive_weight': [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0],
    'temperature': [0.05, 0.07, 0.1, 0.2, 0.5],
    'learning_rate': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
    'kl_weight': [0.01, 0.1, 0.2, 0.5],
    'latent_dim': [128, 256]
}

def run_trial(params):
    """Runs a single training trial with the given parameters."""
    print("\n" + "="*60)
    print(f"🚀 STARTING TRIAL: {params}")
    print("="*60)

    # Construct the training command
    command = [
        'python', 'complete_training.py',
        '--epochs_cvae', str(EPOCHS_PER_TRIAL),
        '--epochs_seg', '0',
        '--contrastive_weight', str(params['contrastive_weight']),
        '--temperature', str(params['temperature']),
        '--learning_rate', str(params['learning_rate']),
        '--kl_weight', str(params['kl_weight']),
        '--latent_dim', str(params['latent_dim'])
    ]

    # Execute the command and capture output
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR during training run:")
        print(e.stderr)
        return None

    # Parse the output to find the best contrastive loss
    best_loss = float('inf')
    for line in output.split('\n'):
        match = re.search(r"best: (\d+\.\d+)", line)
        if match:
            loss_val = float(match.group(1))
            if loss_val < best_loss:
                best_loss = loss_val
    
    if best_loss == float('inf'):
        print("⚠️ Could not parse best contrastive loss from output.")
        return None

    print(f"✅ TRIAL COMPLETE: Best Contrastive Loss = {best_loss:.4f}")
    return {'best_contrastive_loss': best_loss}

def main():
    """Main function to run the hyperparameter sweep."""
    print(f"🔥 Starting Hyperparameter Sweep for {NUM_TRIALS} trials... 🔥")

    for i in range(NUM_TRIALS):
        # Randomly sample parameters
        params = {key: random.choice(values) for key, values in search_space.items()}
        
        # Run the trial
        metrics = run_trial(params)

        # Log the results
        if metrics:
            log_ablation(params, metrics)
            print(f"📊 Logged trial {i+1}/{NUM_TRIALS} to ABLATIONS.md")

    print("\n🎉 Hyperparameter sweep complete! 🎉")
    print("Check ABLATIONS.md for results.")

if __name__ == "__main__":
    main()

