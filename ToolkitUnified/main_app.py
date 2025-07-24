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

        # 1) Configuro il grid del root
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)

        # 2) Posiziono me stesso (il frame principale) con grid
        self.grid(row=0, column=0, sticky="nsew")

        self.style = ttk.Style()
        self.style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'))
        self.create_widgets()

    def create_widgets(self):
        # 3) Configuro il grid del frame principale
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self)
        # posiziono il notebook con grid e sticky “nsew”
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Tab Visualizer
        self.visualizer_frame = ttk.Frame(self.notebook)
        self.visualizer_app = VisualizerApp(self.visualizer_frame)
        # dentro la tab, anche qui abilito grid:
        self.visualizer_frame.columnconfigure(0, weight=1)
        self.visualizer_frame.rowconfigure(0, weight=1)
        self.visualizer_app.grid(row=0, column=0, sticky="nsew")

        # Tab Comparator: stessa cosa
        self.comparator_frame = ttk.Frame(self.notebook)
        self.comparator_app = ComparatorApp(self.comparator_frame)
        self.comparator_frame.columnconfigure(0, weight=1)
        self.comparator_frame.rowconfigure(0, weight=1)
        self.comparator_app.grid(row=0, column=0, sticky="nsew")

        self.notebook.add(self.visualizer_frame, text="🎮 2D Visualizer")
        self.notebook.add(self.comparator_frame, text="📊 Plan Comparator")


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()