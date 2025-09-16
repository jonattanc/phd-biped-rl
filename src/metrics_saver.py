# metrics_saver.py
import csv
import os
from datetime import datetime

def save_complexity_metrics(metrics, circuit_name, output_dir="logs/data"):
    """
    Salva as métricas de complexidade em um arquivo CSV.
    Formato: logs/data/{circuit_name}.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{circuit_name}.csv"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'avg_time', 'std_time', 'success_rate', 'num_episodes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'avg_time': metrics['avg_time'],
            'std_time': metrics['std_time'],
            'success_rate': metrics['success_rate'],
            'num_episodes': len(metrics['total_times'])
        })

    print(f"[INFO] Métricas de complexidade salvas em: {filepath}")
    return filepath