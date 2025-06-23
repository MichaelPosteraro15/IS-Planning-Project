#!/usr/bin/env python3
import re
import csv
import glob
import os

# --- 1) Definisci i regex per ciascuna metrica nel log ---
METRIC_REGEX = {
    'ground_time_ms'   : re.compile(r'grounding time:\s*(\d+)'),
    'f_count'          : re.compile(r'\|f\|:(\d+)'),
    'x_count'          : re.compile(r'\|x\|:(\d+)'),
    'a_count'          : re.compile(r'\|a\|:(\d+)'),
    'p_count'          : re.compile(r'\|p\|:(\d+)'),
    'e_count'          : re.compile(r'\|e\|:(\d+)'),
    'preproc_ms'       : re.compile(r'h1 setup time \(msec\):\s*(\d+)'),
    'search_time_ms'   : re.compile(r'search time \(msec\):\s*(\d+)'),
    'plan_length'      : re.compile(r'plan-length:(\d+)'),
    'plan_cost'        : re.compile(r'metric \(search\):([\d\.]+)'),
    'expanded_nodes'   : re.compile(r'expanded nodes:(\d+)'),
    'evaluated_states' : re.compile(r'states evaluated:(\d+)'),
    'dead_ends'        : re.compile(r'number of dead-ends detected:(\d+)'),
    'duplicates'       : re.compile(r'number of duplicates detected:(\d+)'),
}

# --- 2) Prepara il CSV di output ---
out_csv = 'out.csv'
with open(out_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Intestazione
    header = ['run_name'] + list(METRIC_REGEX.keys())
    writer.writerow(header)

    # --- 3) Per ogni file di log (estensione .log ad esempio) ---
    for filepath in glob.glob('logs/*.log'):
        run_name = os.path.splitext(os.path.basename(filepath))[0]
        text = open(filepath).read()
        row = [run_name]

        # --- 4) Estrai ciascuna metrica col regex corrispondente ---
        for key, regex in METRIC_REGEX.items():
            m = regex.search(text)
            row.append(m.group(1) if m else '')

        # --- 5) Scrivi la riga ---
        writer.writerow(row)

print(f"✅ Summary written to {out_csv}")
