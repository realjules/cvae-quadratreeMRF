
import os
from datetime import datetime

def log_ablation(params, metrics, file_path="../ABLATIONS.md"):
    """
    Logs the parameters and results of an ablation study to a Markdown file.

    Args:
        params (dict): Dictionary of hyperparameters used for the run.
        metrics (dict): Dictionary of results from the run.
        file_path (str): Path to the Markdown log file.
    """
    # Create the file and header if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            header = "| Date | Contrastive Weight | Temperature | Learning Rate | KL Weight | Latent Dim | Best Contrastive Loss |\n"
            header += "|---|---|---|---|---|---|---|"
            f.write(header)

    # Create the new log entry
    log_entry = f"\n| {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    log_entry += f"{params.get('contrastive_weight', 'N/A')} | "
    log_entry += f"{params.get('temperature', 'N/A')} | "
    log_entry += f"{params.get('learning_rate', 'N/A')} | "
    log_entry += f"{params.get('kl_weight', 'N/A')} | "
    log_entry += f"{params.get('latent_dim', 'N/A')} | "
    log_entry += f"{metrics.get('best_contrastive_loss', 'N/A'):.4f} |"

    # Append to the file
    with open(file_path, "a") as f:
        f.write(log_entry)


