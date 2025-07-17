import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os, re, difflib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# --- Data Parsing (unchanged) ---
def parse_all_metrics(content):
    metrics = {}
    patterns = {
        'plan_length': r'plan-length:(\d+)',
        'planning_time': r'planning time \(msec\): (\d+)',
        'search_time': r'search time \(msec\): (\d+)',
        'heuristic_time': r'heuristic time \(msec\): (\d+)',
        'grounding_time': r'grounding time: (\d+)',
        'expanded_nodes': r'expanded nodes:(\d+)',
        'states_evaluated': r'states evaluated:(\d+)',
        'dead_ends': r'number of dead-ends detected:(\d+)',
        'duplicates': r'number of duplicates detected:(\d+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, content, re.IGNORECASE)
        metrics[key] = int(m.group(1)) if m else 0
    return metrics

def parse_plan_actions(content):
    steps = []
    block = re.search(r'found plan:(.*?)(?:plan-length|metric|planning time)', content, re.DOTALL|re.IGNORECASE)
    if block:
        for ln in block.group(1).splitlines():
            ln = ln.strip()
            if not ln or ln.startswith(';'): continue
            ln = re.sub(r'^\d+\.\d+:\s*', '', ln)
            if ln.startswith('(') and ln.endswith(')'):
                steps.append(ln[1:-1].strip())
    if not steps:
        raise ValueError("No valid actions found.")
    metrics = parse_all_metrics(content)
    if metrics['plan_length'] == 0:
        metrics['plan_length'] = len(steps)
    return steps, metrics

# --- Main Application ---
class PlanComparatorApp(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title("Modern PDDL Comparator")
        parent.geometry("1000x750")
        self.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.plan_data = {}
        self.files = []
        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        # Frames
        style.configure('TFrame', background='#eef6fa')
        style.configure('TLabelFrame', background='#eef6fa', relief='flat')
        # Buttons
        style.configure('TButton', font=('Segoe UI', 11), padding=8, background='#007acc', foreground='#ffffff')
        style.map('TButton', background=[('active','#005fa3')])
        # Treeview
        style.configure('Treeview', background='#ffffff', fieldbackground='#f0f8ff', rowheight=30, font=('Consolas', 10))
        style.configure('Treeview.Heading', font=('Segoe UI Semibold', 11), background='#007acc', foreground='#ffffff')
        style.layout('Treeview', [('Treeview.treearea', {'sticky': 'nswe'})])

    def _build_ui(self):
        lf = ttk.LabelFrame(self, text="Select Plans")
        lf.pack(fill=tk.X, pady=10)
        for i, label in enumerate(["Classic", "Numeric"]):
            btn = ttk.Button(lf, text=f"Load {label}", command=lambda idx=i+1: self.load_plan(idx))
            btn.grid(row=i, column=0, padx=5, pady=5)
            lbl = ttk.Label(lf, text="No file", foreground='#d9534f')
            lbl.grid(row=i, column=1, padx=5, pady=5, sticky=tk.W)
            setattr(self, f'label_{i+1}', lbl)
        self.compare_btn = ttk.Button(self, text="Compare Plans", command=self.compare_plans)
        self.compare_btn.pack(fill=tk.X, pady=5)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self._create_summary_tab()
        self._create_graph_tab()
        self._create_diff_tab()

    def _create_summary_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Summary")
        columns = ('metric', 'plan1', 'plan2')
        tree = ttk.Treeview(tab, columns=columns, show='headings')
        # configure headings and widths
        tree.heading('metric', text='Metric', anchor=tk.W)
        tree.column('metric', anchor=tk.W, width=200)
        tree.heading('plan1', text='Plan A', anchor=tk.CENTER)
        tree.column('plan1', anchor=tk.CENTER, width=150)
        tree.heading('plan2', text='Plan B', anchor=tk.CENTER)
        tree.column('plan2', anchor=tk.CENTER, width=150)
        # alternating row colors
        tree.tag_configure('odd', background='#fcfcfc')
        tree.tag_configure('even', background='#f0f8ff')
        # add vertical scrollbar
        vsb = ttk.Scrollbar(tab, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree_summary = tree

    def _create_graph_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Performance Graph")
        self.fig, self.ax = plt.subplots(figsize=(8,6))
        self.canvas = FigureCanvasTkAgg(self.fig, tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_diff_tab(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Side-by-Side Diff")
        f = ttk.Frame(tab)
        f.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.txt1 = tk.Text(f, font=('Consolas',10), wrap='none')
        self.txt2 = tk.Text(f, font=('Consolas',10), wrap='none')
        scroll = ttk.Scrollbar(f, orient='vertical', command=self._sync_scroll)
        self.txt1['yscrollcommand'] = scroll.set
        self.txt2['yscrollcommand'] = scroll.set
        self.txt1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        for txt, tag, bg in [(self.txt1,'delete','#fde2e2'),(self.txt2,'insert','#e2fde2')]:
            txt.tag_config(tag, background=bg)

    def _sync_scroll(self, *args):
        self.txt1.yview(*args)
        self.txt2.yview(*args)

    def load_plan(self, idx):
        path = filedialog.askopenfilename(filetypes=[('Text','*.txt')])
        if not path: return
        with open(path) as f: text = f.read()
        try:
            actions, metrics = parse_plan_actions(text)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self.plan_data[idx] = {'actions':actions, 'metrics':metrics}
        lbl = getattr(self, f'label_{idx}')
        lbl.config(text=os.path.basename(path), foreground='#28a745')
        self.files.append(os.path.basename(path))

    def compare_plans(self):
        if 1 not in self.plan_data or 2 not in self.plan_data:
            messagebox.showwarning("Load both files")
            return
        self._update_summary(); self._update_graph(); self._update_diff()

    def _update_summary(self):
        m1 = self.plan_data[1]['metrics']; m2 = self.plan_data[2]['metrics']
        self.tree_summary.delete(*self.tree_summary.get_children())
        mapping = {
            'plan_length':'Plan Length', 'planning_time':'Planning Time (ms)',
            'search_time':'Search Time (ms)', 'heuristic_time':'Heuristic Time (ms)',
            'grounding_time':'Grounding Time (ms)', 'expanded_nodes':'Expanded Nodes',
            'states_evaluated':'States Evaluated','dead_ends':'Dead-ends','duplicates':'Duplicates'
        }
        for idx, (key, label) in enumerate(mapping.items()):
            tag = 'odd' if idx % 2 == 0 else 'even'
            self.tree_summary.insert('', 'end', values=(label, m1[key], m2[key]), tags=(tag,))

    def _update_graph(self):
        m1 = self.plan_data[1]['metrics']; m2 = self.plan_data[2]['metrics']
        self.ax.clear()
        labels = ['Length','Plan Time','Expanded','Heuristic','States','Grounding']
        keys = ['plan_length','planning_time','expanded_nodes','heuristic_time','states_evaluated','grounding_time']
        x = np.arange(len(labels)); w=0.35
        r1 = self.ax.bar(x-w/2, [m1[k] for k in keys], w, label=self.files[0])
        r2 = self.ax.bar(x+w/2, [m2[k] for k in keys], w, label=self.files[1])
        # add value labels
        self.ax.bar_label(r1, padding=3, rotation=90, fontsize=8)
        self.ax.bar_label(r2, padding=3, rotation=90, fontsize=8)
        self.ax.set_xticks(x); self.ax.set_xticklabels(labels, rotation=30)
        self.ax.set_ylabel('Values'); self.ax.legend(); self.fig.tight_layout()
        self.canvas.draw()

    def _update_diff(self):
        a1 = self.plan_data[1]['actions']; a2 = self.plan_data[2]['actions']
        self.txt1.delete('1.0', tk.END); self.txt2.delete('1.0', tk.END)
        sm = difflib.SequenceMatcher(None, a1, a2)
        l1 = l2 = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
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
                diff = abs((i2 - i1) - (j2 - j1))
                for _ in range(diff):
                    if (i2 - i1) < (j2 - j1):
                        self.txt1.insert(tk.END, '\n')
                    else:
                        self.txt2.insert(tk.END, '\n')
        self.txt1.config(state='disabled'); self.txt2.config(state='disabled')

if __name__ == '__main__':
    root = tk.Tk()
    app = PlanComparatorApp(root)
    root.mainloop()

'''
In the "Side-by-Side Diff" tab (third section):

Red lines (tag "delete"): indicate actions that are present in Plan A (left column) but not 
in Plan B. These are therefore steps "deleted" with respect to the other plan.

Green lines (tag "insert"): indicate actions that appear in Plan B (right column) but not 
in Plan A. These are therefore steps "inserted" with respect to the other plan.

Lines without color (white) represent identical actions in both plans, kept "equal." 
This way, you can immediately see which steps vary between the respective outputs.
'''