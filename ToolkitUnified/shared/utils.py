import os
import csv
from datetime import datetime

def save_metrics(metrics, filename="metrics.csv"):
    os.makedirs('data', exist_ok=True)
    path = os.path.join('data', filename)
    file_exists = os.path.isfile(path)
    
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def get_timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')