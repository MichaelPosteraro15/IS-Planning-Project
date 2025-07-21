import tkinter as tk
from tkinter import filedialog, messagebox
import csv

def show_metrics_popup(metrics):
    window = tk.Toplevel()
    window.title(f"📊 Metrics - {metrics.get('run_name', 'Unknown')}")
    window.geometry("600x750")
    
    header_frame = tk.Frame(window, bg='#2E7D32', height=60)
    header_frame.pack(fill=tk.X, padx=0, pady=0)
    header_frame.pack_propagate(False)
    
    title_label = tk.Label(header_frame, text="🎯 Snowman Planner Metrics", 
                         font=('Arial', 16, 'bold'), fg='white', bg='#2E7D32')
    title_label.pack(pady=15)
    
    content_frame = tk.Frame(window, bg='#f5f5f5')
    content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
    
    text = tk.Text(content_frame, wrap=tk.WORD, font=('Consolas', 11), 
                  bg='#ffffff', fg='#333333', relief='sunken', bd=1)
    scrollbar = tk.Scrollbar(content_frame, orient='vertical', command=text.yview)
    text.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    execution_metrics = {
        'Run Name': metrics.get('run_name', 'N/A'),
        'Execution Time (ms)': metrics.get('execution_time_ms', 0),
        'Timestamp': metrics.get('timestamp', 'N/A')
    }
    
    plan_metrics = {
        'Plan Length': metrics.get('plan_length', 0),
        'Total Cost': metrics.get('total_cost', 0),
        'Goal Count': metrics.get('goal_count', 0)
    }
    
    action_metrics = {
        'Move Character Count': metrics.get('move_character_count', 0),
        'Move Ball Count': metrics.get('move_ball_count', 0),
        'Ball Growth Count': metrics.get('ball_growth_count', 0)
    }
    
    final_state = {
        'Final Ball Locations': metrics.get('final_ball_locations', 'N/A'),
        'Final Ball Sizes': metrics.get('final_ball_sizes', 'N/A')
    }
    
    text.insert(tk.END, f"{'='*60}\n")
    text.insert(tk.END, f"SNOWMAN PLANNER EXECUTION METRICS\n")
    text.insert(tk.END, f"{'='*60}\n\n")
    
    for category, data in [
        ('EXECUTION INFO', execution_metrics),
        ('PLAN METRICS', plan_metrics),
        ('ACTION BREAKDOWN', action_metrics),
        ('FINAL STATE', final_state)
    ]:
        text.insert(tk.END, f"{category}\n")
        text.insert(tk.END, f"{'-'*40}\n")
        for key, value in data.items():
            text.insert(tk.END, f"{key:.<30} {value}\n")
        text.insert(tk.END, f"\n")
    
    text.configure(state='disabled')
    
    button_frame = tk.Frame(window, bg='#f5f5f5')
    button_frame.pack(fill=tk.X, padx=15, pady=10)
    
    export_button = tk.Button(button_frame, text="Export CSV", 
                            command=lambda: export_metrics_csv(metrics),
                            font=('Arial', 10, 'bold'), bg='#388E3C', fg='white')
    export_button.pack(side=tk.LEFT, padx=(0, 10))
    
    close_button = tk.Button(button_frame, text="Close", 
                           command=window.destroy,
                           font=('Arial', 10, 'bold'), bg='#1976D2', fg='white')
    close_button.pack(side=tk.RIGHT)

def export_metrics_csv(metrics):
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save Metrics As"
        )
        
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
                writer.writeheader()
                writer.writerow(metrics)
            
            messagebox.showinfo("Export Successful", f"Metrics exported to:\n{file_path}")
    except Exception as e:
        messagebox.showerror("Export Error", f"Failed to export metrics:\n{str(e)}")