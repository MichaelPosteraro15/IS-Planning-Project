import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import difflib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from .core import parse_plan_actions

class ComparatorApp(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.plan_data = {}
        self.files = []
        self.create_widgets()
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Segoe UI', 10))
        self.style.configure('Treeview', font=('Consolas', 10), rowheight=25)
        self.style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(control_frame, text="Load Plan A", 
                  command=lambda: self.load_plan(1)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(control_frame, text="Load Plan B", 
                  command=lambda: self.load_plan(2)).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Compare Plans", 
                  command=self.compare_plans).grid(row=0, column=2, padx=5, pady=5)
        
        self.label1 = ttk.Label(control_frame, text="No file", foreground='red')
        self.label1.grid(row=0, column=3, padx=10, pady=5)
        self.label2 = ttk.Label(control_frame, text="No file", foreground='red')
        self.label2.grid(row=0, column=4, padx=10, pady=5)
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.create_summary_tab()
        self.create_graph_tab()
        self.create_diff_tab()

    def create_summary_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Summary")
        
        columns = ('metric', 'plan1', 'plan2')
        self.tree_summary = ttk.Treeview(tab, columns=columns, show='headings')
        
        self.tree_summary.heading('metric', text='Metric')
        self.tree_summary.column('metric', width=200, anchor=tk.W)
        
        self.tree_summary.heading('plan1', text='Plan A')
        self.tree_summary.column('plan1', width=150, anchor=tk.CENTER)
        
        self.tree_summary.heading('plan2', text='Plan B')
        self.tree_summary.column('plan2', width=150, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=self.tree_summary.yview)
        self.tree_summary.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_summary.pack(fill=tk.BOTH, expand=True)

    def create_graph_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Performance")
        
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_diff_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Action Diff")
        
        frame = ttk.Frame(tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.txt1 = tk.Text(frame, font=('Consolas', 10), wrap='none', bg='#f8f8f8')
        self.txt2 = tk.Text(frame, font=('Consolas', 10), wrap='none', bg='#f8f8f8')
        
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.sync_scroll)
        self.txt1['yscrollcommand'] = scrollbar.set
        self.txt2['yscrollcommand'] = scrollbar.set
        
        self.txt1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.txt1.tag_config('delete', background='#ffdddd')
        self.txt2.tag_config('insert', background='#ddffdd')

    def sync_scroll(self, *args):
        self.txt1.yview(*args)
        self.txt2.yview(*args)

    def load_plan(self, idx):
        path = filedialog.askopenfilename(filetypes=[('Text','*.txt')])
        if not path: 
            return
        
        try:
            with open(path) as f: 
                text = f.read()
            actions, metrics = parse_plan_actions(text)
            self.plan_data[idx] = {'actions': actions, 'metrics': metrics}
            
            label = getattr(self, f'label{idx}')
            label.config(text=os.path.basename(path), foreground='green')
            self.files.append(os.path.basename(path))
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def compare_plans(self):
        if 1 not in self.plan_data or 2 not in self.plan_data:
            messagebox.showwarning("Missing Data", "Please load both plans first")
            return
            
        self.update_summary()
        self.update_graph()
        self.update_diff()

    def update_summary(self):
        self.tree_summary.delete(*self.tree_summary.get_children())
        m1 = self.plan_data[1]['metrics']
        m2 = self.plan_data[2]['metrics']
        
        metrics_map = {
            'plan_length': 'Plan Length',
            'planning_time': 'Planning Time (ms)',
            'search_time': 'Search Time (ms)',
            'heuristic_time': 'Heuristic Time (ms)',
            'grounding_time': 'Grounding Time (ms)',
            'expanded_nodes': 'Expanded Nodes',
            'states_evaluated': 'States Evaluated',
            'dead_ends': 'Dead-ends',
            'duplicates': 'Duplicates'
        }
        
        for idx, (key, label) in enumerate(metrics_map.items()):
            self.tree_summary.insert('', 'end', values=(
                label, 
                m1.get(key, 0), 
                m2.get(key, 0)
            ))

    def update_graph(self):
        self.ax.clear()
        m1 = self.plan_data[1]['metrics']
        m2 = self.plan_data[2]['metrics']
        
        labels = ['Length', 'Plan Time', 'Expanded', 'Heuristic', 'States', 'Grounding']
        keys = ['plan_length', 'planning_time', 'expanded_nodes', 'heuristic_time', 'states_evaluated', 'grounding_time']
        
        values1 = [m1.get(k, 0) for k in keys]
        values2 = [m2.get(k, 0) for k in keys]
        
        x = np.arange(len(labels))
        width = 0.35
        
        self.ax.bar(x - width/2, values1, width, label=self.files[0])
        self.ax.bar(x + width/2, values2, width, label=self.files[1])
        
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels)
        self.ax.legend()
        self.ax.set_title('Plan Comparison Metrics')
        
        self.canvas.draw()

    def update_diff(self):
        a1 = self.plan_data[1]['actions']
        a2 = self.plan_data[2]['actions']
        
        self.txt1.config(state='normal')
        self.txt2.config(state='normal')
        self.txt1.delete('1.0', tk.END)
        self.txt2.delete('1.0', tk.END)
        
        diff = difflib.SequenceMatcher(None, a1, a2)
        l1 = l2 = 0
        
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            if tag == 'equal':
                for i in range(i1, i2):
                    l1 += 1; l2 += 1
                    self.txt1.insert(tk.END, f"{l1:02d}: {a1[i]}\n")
                    self.txt2.insert(tk.END, f"{l2:02d}: {a1[i]}\n")
            else:
                if tag in ('delete', 'replace'):
                    for act in a1[i1:i2]:
                        l1 += 1
                        self.txt1.insert(tk.END, f"{l1:02d}: {act}\n", 'delete')
                if tag in ('insert', 'replace'):
                    for act in a2[j1:j2]:
                        l2 += 1
                        self.txt2.insert(tk.END, f"{l2:02d}: {act}\n", 'insert')
        
        self.txt1.config(state='disabled')
        self.txt2.config(state='disabled')