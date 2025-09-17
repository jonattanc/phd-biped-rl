# utils.py
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_PATH = os.path.join(PROJECT_ROOT, "tmp")
TEMPORARY_FOLDERS = [TMP_PATH, "logs", "logs/data", "logs/data/models", "logs/tensorboard"]
