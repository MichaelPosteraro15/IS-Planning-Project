import tkinter as tk
from tkinter import ttk
from visualizer.ui import VisualizerApp
from comparator.ui import ComparatorApp

class MainApplication(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.parent.title("Snowman Planner Toolkit")
        self.parent.geometry("800x600")
        self.style = ttk.Style()
        self.style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'))
        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        
        # Tab Visualizer
        self.visualizer_frame = ttk.Frame(self.notebook)
        self.visualizer_app = VisualizerApp(self.visualizer_frame)
        self.visualizer_app.pack(fill=tk.BOTH, expand=True)
        
        # Tab Comparator
        self.comparator_frame = ttk.Frame(self.notebook)
        self.comparator_app = ComparatorApp(self.comparator_frame)
        self.comparator_app.pack(fill=tk.BOTH, expand=True)
        
        self.notebook.add(self.visualizer_frame, text="🎮 2D Visualizer")
        self.notebook.add(self.comparator_frame, text="📊 Plan Comparator")
        
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()