# logger.py
import csv
import os
from datetime import datetime


class EpisodeLogger:
    def __init__(self, output_dir="logs/data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.filename = os.path.join(output_dir, f"episodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.headers = ["circuito", "avatar", "papel", "semente", "repeticao", "tempo_total", "sucesso", "hiperparametros"]
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_episode(self, circuito, avatar, papel, semente, repeticao, tempo_total, sucesso, hiperparametros=""):
        with open(self.filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([circuito, avatar, papel, semente, repeticao, tempo_total, sucesso, hiperparametros])
