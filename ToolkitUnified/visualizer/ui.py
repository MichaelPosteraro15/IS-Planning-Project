import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import Button
from .core import *
from .metrics import show_metrics_popup
import time

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from .core import *
from .metrics import show_metrics_popup
import time

class VisualizerApp(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.selected_problem_file = None
        self.selected_plan_file = None
        self.current_metrics = {}
        self.visualization_completed = False
        self.animation_running = False
        self.frames = []
        self.current_frame = 0
        self.paused = True
        self.ani = None
        self.metrics_calculator = MetricsCalculator()
        
        # Modern color scheme (aggiornato)
        self.colors = {
            'primary':        '#00695C',  # Teal scuro
            'secondary':      '#0288D1',  # Blu medio
            'accent':         '#FFB300',  # Giallo ambra
            'success':        '#2E7D32',  # Verde scuro
            'warning':        '#F57C00',  # Arancio intenso
            'error':          '#D32F2F',  # Rosso scuro
            'background':     '#F5F5F5',
            'surface':        '#FFFFFF',
            'text_primary':   '#212121',
            'text_secondary': '#616161',
        }
        
        self.configure_styles()
        self.create_widgets()
        self.reset_ui()

    def configure_styles(self):
        """Configure modern ttk styles"""
        style = ttk.Style()
        style.configure('Modern.TButton',
                        borderwidth=0,
                        focuscolor='none',
                        relief='flat',
                        padding=(12, 8))
        style.configure('Accent.TButton',
                        borderwidth=0,
                        focuscolor='none',
                        relief='flat',
                        padding=(12, 8))
        style.configure('Card.TFrame',
                        relief='flat',
                        borderwidth=1,
                        background=self.colors['surface'])
        style.configure('Header.TLabel',
                        foreground=self.colors['text_primary'],
                        font=('Segoe UI', 10, 'bold'))
        style.configure('Status.TLabel',
                        foreground=self.colors['text_secondary'],
                        font=('Segoe UI', 9))

    def create_widgets(self):
        self.configure(bg=self.colors['background'])
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor(self.colors['surface'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(highlightthickness=0)
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(10,5))
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.configure(bg=self.colors['surface'])
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0,5))
        self.step_text_artist = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='sans-serif',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['surface'], 
                      alpha=0.9, edgecolor=self.colors['primary'], linewidth=1)
        )
        self.create_header_section()
        self.create_file_selection_section()
        self.create_control_section()

    def create_header_section(self):
        header_frame = tk.Frame(self, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, padx=10, pady=(10,0))
        tk.Label(header_frame, 
                 text="🎯 Snowman Planner Visualizer",
                 font=('Segoe UI', 16, 'bold'),
                 fg=self.colors['primary'],
                 bg=self.colors['background']).pack(side=tk.LEFT)
        self.status_frame = tk.Frame(header_frame, bg=self.colors['background'])
        self.status_frame.pack(side=tk.RIGHT)
        self.status_indicator = tk.Label(self.status_frame,
                                         text="●",
                                         font=('Segoe UI', 12),
                                         fg=self.colors['text_secondary'],
                                         bg=self.colors['background'])
        self.status_indicator.pack(side=tk.RIGHT, padx=(0,5))
        self.status_label = tk.Label(self.status_frame,
                                     text="Ready",
                                     font=('Segoe UI', 9),
                                     fg=self.colors['text_secondary'],
                                     bg=self.colors['background'])
        self.status_label.pack(side=tk.RIGHT)

    def create_file_selection_section(self):
        file_card = tk.Frame(self, bg=self.colors['surface'], relief='solid', bd=1)
        file_card.pack(fill=tk.X, padx=10, pady=5)
        card_header = tk.Frame(file_card, bg=self.colors['surface'])
        card_header.pack(fill=tk.X, padx=15, pady=(10,5))
        tk.Label(card_header, 
                 text="📁 File Selection",
                 font=('Segoe UI', 11, 'bold'),
                 fg=self.colors['text_primary'],
                 bg=self.colors['surface']).pack(side=tk.LEFT)
        button_frame = tk.Frame(file_card, bg=self.colors['surface'])
        button_frame.pack(fill=tk.X, padx=15, pady=5)
        self.problem_btn = tk.Button(button_frame, 
                                     text="📋 Select Problem",
                                     command=self.select_problem_file,
                                     font=('Segoe UI', 9),
                                     fg='white',
                                     bg=self.colors['primary'],
                                     activebackground=self.colors['secondary'],
                                     relief='flat',
                                     cursor='hand2',
                                     padx=15, pady=8)
        self.problem_btn.pack(side=tk.LEFT, padx=(0,10))
        self.plan_btn = tk.Button(button_frame, 
                                  text="📝 Select Plan",
                                  command=self.select_plan_file,
                                  font=('Segoe UI', 9),
                                  fg='white',
                                  bg=self.colors['primary'],
                                  activebackground=self.colors['secondary'],
                                  relief='flat',
                                  cursor='hand2',
                                  padx=15, pady=8)
        self.plan_btn.pack(side=tk.LEFT, padx=(0,10))
        self.load_btn = tk.Button(button_frame, 
                                  text="🚀 Load Files",
                                  command=self.load_files,
                                  font=('Segoe UI', 9, 'bold'),
                                  fg='white',
                                  bg=self.colors['accent'],
                                  activebackground='#FFA000',
                                  relief='flat',
                                  cursor='hand2',
                                  padx=20, pady=8)
        self.load_btn.pack(side=tk.LEFT)
        status_frame = tk.Frame(file_card, bg=self.colors['surface'])
        status_frame.pack(fill=tk.X, padx=15, pady=(5,10))
        self.problem_label = tk.Label(status_frame, 
                                      text="Problem: Not selected",
                                      font=('Segoe UI', 9),
                                      fg=self.colors['text_secondary'],
                                      bg=self.colors['surface'])
        self.problem_label.pack(anchor=tk.W)
        self.plan_label = tk.Label(status_frame, 
                                   text="Plan: Not selected",
                                   font=('Segoe UI', 9),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['surface'])
        self.plan_label.pack(anchor=tk.W)

    def create_control_section(self):
        control_card = tk.Frame(self, bg=self.colors['surface'], relief='solid', bd=1)
        control_card.pack(fill=tk.X, padx=10, pady=5)
        card_header = tk.Frame(control_card, bg=self.colors['surface'])
        card_header.pack(fill=tk.X, padx=15, pady=(10,5))
        tk.Label(card_header, 
                 text="🎮 Animation Controls",
                 font=('Segoe UI', 11, 'bold'),
                 fg=self.colors['text_primary'],
                 bg=self.colors['surface']).pack(side=tk.LEFT)
        button_frame = tk.Frame(control_card, bg=self.colors['surface'])
        button_frame.pack(fill=tk.X, padx=15, pady=5)
        left_controls = tk.Frame(button_frame, bg=self.colors['surface'])
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(left_controls, 
                  text="⏮️ Step Back",
                  command=self.step_backward,
                  font=('Segoe UI', 9),
                  fg='white',
                  bg=self.colors['secondary'],
                  activebackground='#039BE5',
                  relief='flat',
                  cursor='hand2',
                  padx=12, pady=8).pack(side=tk.LEFT, padx=(0,5))
        self.toggle_btn = tk.Button(left_controls, 
                                   text="▶️ Play",
                                   command=self.toggle_animation,
                                   font=('Segoe UI', 9, 'bold'),
                                   fg='white',
                                   bg=self.colors['success'],
                                   activebackground='#1B5E20',
                                   relief='flat',
                                   cursor='hand2',
                                   padx=20, pady=8)
        self.toggle_btn.pack(side=tk.LEFT, padx=(0,8))
        tk.Button(left_controls, 
                  text="⏭️ Step Forward",
                  command=self.step_forward,
                  font=('Segoe UI', 9),
                  fg='white',
                  bg=self.colors['secondary'],
                  activebackground='#039BE5',
                  relief='flat',
                  cursor='hand2',
                  padx=12, pady=8).pack(side=tk.LEFT, padx=(0,5))
        tk.Button(left_controls, 
                  text="🔄 Restart",
                  command=self.restart_animation,
                  font=('Segoe UI', 9),
                  fg='white',
                  bg=self.colors['warning'],
                  activebackground='#E65100',
                  relief='flat',
                  cursor='hand2',
                  padx=12, pady=8).pack(side=tk.LEFT, padx=(0,5))
        right_controls = tk.Frame(button_frame, bg=self.colors['surface'])
        right_controls.pack(side=tk.RIGHT)
        
        tk.Button(right_controls, 
                  text="⚙️ Settings",
                  command=self.show_settings_menu,
                  font=('Segoe UI', 9),
                  fg='white',
                  bg=self.colors['primary'],
                  activebackground=self.colors['secondary'],
                  relief='flat',
                  cursor='hand2',
                  padx=15, pady=8).pack(side=tk.RIGHT, padx=(5,0))
        speed_frame = tk.Frame(control_card, bg=self.colors['surface'])
        speed_frame.pack(fill=tk.X, padx=15, pady=(5,10))
        tk.Label(speed_frame, 
                 text="⚡ Animation Speed",
                 font=('Segoe UI', 9, 'bold'),
                 fg=self.colors['text_primary'],
                 bg=self.colors['surface']).pack(side=tk.LEFT)
        self.speed_slider = tk.Scale(speed_frame, 
                                     from_=0, to=100, 
                                     orient=tk.HORIZONTAL,
                                     command=self.update_animation_speed,
                                     font=('Segoe UI', 8),
                                     fg=self.colors['text_primary'],
                                     bg=self.colors['surface'],
                                     activebackground=self.colors['primary'],
                                     highlightthickness=0,
                                     length=200)
        self.speed_slider.set(50)
        self.speed_slider.pack(side=tk.RIGHT, padx=(10,0))

    # ... il resto dei metodi rimane invariato ...


    def update_status(self, message, status_type="info"):
        """Update status indicator with modern styling"""
        colors = {
            "info": self.colors['text_secondary'],
            "success": self.colors['success'],
            "warning": self.colors['warning'],
            "error": self.colors['error']
        }
        
        self.status_label.config(text=message)
        self.status_indicator.config(fg=colors.get(status_type, self.colors['text_secondary']))

    def reset_ui(self):
        self.frames = []
        self.current_metrics = {}
        self.visualization_completed = False
        self.animation_running = False
        
        if self.ani is not None:
            try:
                self.ani.event_source.stop()
            except:
                pass
            self.ani = None
        
        self.current_frame = 0
        self.paused = True
        self.toggle_btn.config(text='▶️ Play')
        
        # Modern welcome screen
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        
        # Gradient background effect
        self.ax.add_patch(plt.Rectangle((0, 0), 1, 1, 
                                       facecolor=self.colors['primary'], 
                                       alpha=0.05))
        
        # Modern welcome text
        self.ax.text(0.5, 0.65, 'Snowman Planner Visualizer', 
                    ha='center', va='center', fontsize=20, fontweight='bold', 
                    color=self.colors['primary'], fontfamily='sans-serif')
        
        self.ax.text(0.5, 0.52, 'Select Problem and Plan files, then click "Load Files"', 
                    ha='center', va='center', fontsize=12, 
                    color=self.colors['text_primary'], fontfamily='sans-serif')
        
        self.ax.text(0.5, 0.42, 'Use the animation controls to navigate through the visualization', 
                    ha='center', va='center', fontsize=10, style='italic', 
                    color=self.colors['text_secondary'], fontfamily='sans-serif')
        
        # Feature highlights
        features = [
            "Real-time metrics analysis",
            "Step-by-step visualization", 
            "Export capabilities",
            "Variable speed control"
        ]
        
        for i, feature in enumerate(features):
            self.ax.text(0.5, 0.28 - i*0.04, feature, 
                        ha='center', va='center', fontsize=9, 
                        color=self.colors['secondary'], fontfamily='sans-serif')
        
        # Reset file status
        self.problem_label.config(text="Problem: Not selected", 
                                 fg=self.colors['text_secondary'])
        self.plan_label.config(text="Plan: Not selected", 
                              fg=self.colors['text_secondary'])
        
        self.fig.suptitle("", fontsize=16, fontweight='bold')
        self.canvas.draw()
        
        self.update_status("Ready", "info")

    def select_problem_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PDDL files", "*.pddl")],
            title="Select Problem File"
        )
        if file_path:
            self.selected_problem_file = file_path
            filename = os.path.basename(file_path)
            self.problem_label.config(text=f"Problem: {filename}", 
                                     fg=self.colors['success'])
            self.update_status(f"Problem file selected: {filename}", "success")

    def select_plan_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt")],
            title="Select Plan File"
        )
        if file_path:
            self.selected_plan_file = file_path
            filename = os.path.basename(file_path)
            self.plan_label.config(text=f"Plan: {filename}", 
                                  fg=self.colors['success'])
            self.update_status(f"Plan file selected: {filename}", "success")

    def load_files(self):
        if not self.selected_problem_file or not self.selected_plan_file:
            messagebox.showerror("Error", "Both problem and plan files must be selected")
            self.update_status("Error: Missing files", "error")
            return False
            
        try:
            self.update_status("Loading files...", "info")
            self.reset_ui()
            self.metrics_calculator.start_timing()
            
            problem = parse_problem(self.selected_problem_file)
            plan = parse_plan(self.selected_plan_file)
            
            with open(self.selected_problem_file, 'r') as f:
                content = f.read()
                domain_match = re.search(r'\(:domain (\S+)\)', content)
                problem['domain'] = domain_match.group(1) if domain_match else 'unknown'
            
            self.frames = build_frames(problem, plan)
            self.current_frame = 0
            self.paused = True
            
            plan_name = os.path.splitext(os.path.basename(self.selected_plan_file))[0]
            self.current_metrics = self.metrics_calculator.finalize_metrics(
                {'balls': self.frames[-1]['balls'], 'ball_size': self.frames[-1]['ball_size']}, 
                plan_name
            )
            
            save_metrics_to_csv(self.current_metrics)
            
            if self.frames:
                draw(self.ax, self.frames[0], self.step_text_artist)
                self.fig.suptitle("Snowman Planner Visualizer - Ready to Play", 
                                 fontsize=14, fontweight='bold', color=self.colors['primary'])
                self.canvas.draw()
                
            self.update_status("Files loaded successfully!", "success")
            messagebox.showinfo("Success", "Files loaded successfully!")
            return True
            
        except Exception as e:
            error_msg = f"Error loading files: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.update_status("Error loading files", "error")
            self.reset_ui()
            return False
        finally:
            self.metrics_calculator.end_timing()

    def animate(self, frame_num):
        if not self.frames or self.paused:
            return
            
        if frame_num >= len(self.frames):
            if not self.visualization_completed:
                self.visualization_completed = True
                self.paused = True
                self.toggle_btn.config(text='▶️ Play')
                self.animation_running = False
                self.update_status("Animation completed", "success")
                if self.current_metrics:
                    show_metrics_popup(self.current_metrics)
            return
            
        self.current_frame = frame_num
        draw(self.ax, self.frames[frame_num], self.step_text_artist)
        
        progress = (frame_num) / max(len(self.frames), 1) * 100
        self.fig.suptitle(f"Snowman Planner Visualizer - Progress: {progress:.1f}%", 
                         fontsize=14, fontweight='bold', color=self.colors['primary'])
        self.canvas.draw()
        self.animation_running = True
        self.update_status(f"Playing animation - {progress:.1f}%", "info")

    def toggle_animation(self):
        if not self.frames:
            messagebox.showwarning("No Animation", "Please load files first to start animation.")
            self.update_status("No animation loaded", "warning")
            return
            
        self.paused = not self.paused
        
        if self.paused:
            self.toggle_btn.config(text='▶️ Play')
            self.animation_running = False
            self.update_status("Animation paused", "info")
        else:
            self.toggle_btn.config(text='⏸️ Pause')
            self.animation_running = True
            self.update_status("Animation playing", "info")
            
            if self.visualization_completed or self.current_frame >= len(self.frames):
                self.current_frame = 0
                self.visualization_completed = False
            
            if self.ani is not None:
                try:
                    self.ani.event_source.stop()
                except:
                    pass
            
            self.ani = FuncAnimation(
                self.fig, self.animate, 
                frames=range(self.current_frame, len(self.frames)), 
                interval=int(PLT_PAUSE * 1000),
                repeat=False
            )
        self.canvas.draw()

    def step_forward(self):
        if not self.frames:
            self.update_status("No animation loaded", "warning")
            return
            
        if self.current_frame < len(self.frames) - 1:
            self.current_frame += 1
            draw(self.ax, self.frames[self.current_frame], self.step_text_artist)
            progress = (self.current_frame) / max(len(self.frames), 1) * 100
            self.fig.suptitle(f"Snowman Planner Visualizer - Progress: {progress:.1f}%", 
                             fontsize=14, fontweight='bold', color=self.colors['primary'])
            self.canvas.draw()
            self.update_status(f"Step {self.current_frame + 1}/{len(self.frames)}", "info")

    def step_backward(self):
        if not self.frames:
            self.update_status("No animation loaded", "warning")
            return
            
        if self.current_frame > 0:
            self.current_frame -= 1
            draw(self.ax, self.frames[self.current_frame], self.step_text_artist)
            progress = (self.current_frame) / max(len(self.frames), 1) * 100
            self.fig.suptitle(f"Snowman Planner Visualizer - Progress: {progress:.1f}%", 
                             fontsize=14, fontweight='bold', color=self.colors['primary'])
            self.canvas.draw()
            self.update_status(f"Step {self.current_frame + 1}/{len(self.frames)}", "info")

    def restart_animation(self):
        if not self.frames:
            self.update_status("No animation loaded", "warning")
            return
            
        self.current_frame = 0
        self.paused = True
        self.visualization_completed = False
        self.animation_running = False
        self.toggle_btn.config(text='▶️ Play')
        
        if self.ani is not None:
            try:
                self.ani.event_source.stop()
            except:
                pass
        
        draw(self.ax, self.frames[0], self.step_text_artist)
        self.fig.suptitle("Snowman Planner Visualizer - Ready to Play", 
                         fontsize=14, fontweight='bold', color=self.colors['primary'])
        self.canvas.draw()
        self.update_status("Animation restarted", "info")

    def update_animation_speed(self, val):
        global PLT_PAUSE
        PLT_PAUSE = 0.001 + (100 - int(val)) * 0.001
        self.update_status(f"Speed: {val}%", "info")

    def show_metrics(self):
        if not self.current_metrics:
            messagebox.showwarning("No Metrics", "No metrics available. Please load and run a visualization first.")
            self.update_status("No metrics available", "warning")
            return
        show_metrics_popup(self.current_metrics)

    def show_settings_menu(self):
        settings_window = tk.Toplevel(self)
        settings_window.title("⚙️ Settings")
        settings_window.geometry("320x400")
        settings_window.configure(bg=self.colors['background'])
        settings_window.resizable(False, False)
        
        # Center window
        settings_window.transient(self.parent)
        settings_window.grab_set()
        
        # Header
        header = tk.Frame(settings_window, bg=self.colors['primary'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, 
                text="⚙️ Settings", 
                font=('Segoe UI', 14, 'bold'),
                fg='white',
                bg=self.colors['primary']).pack(expand=True)
        
        # Content area
        content = tk.Frame(settings_window, bg=self.colors['background'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Modern buttons
        buttons = [
            #("📊 View Metrics", self.show_metrics),
            ("🔄 Restart Animation", self.restart_animation),
            ("🔧 Reset Application", self.reset_ui),
            ("❓ Show Help", self.show_help),
            ("ℹ️ About", self.show_about)
        ]
        
        for text, command in buttons:
            btn = tk.Button(content, 
                           text=text,
                           command=command,
                           font=('Segoe UI', 10),
                           fg='white',
                           bg=self.colors['primary'],
                           activebackground=self.colors['success'],
                           relief='flat',
                           cursor='hand2',
                           pady=10)
            btn.pack(fill=tk.X, pady=5)
        
        # Close button
        close_btn = tk.Button(content, 
                             text="✅ Close",
                             command=settings_window.destroy,
                             font=('Segoe UI', 10, 'bold'),
                             fg='white',
                             bg=self.colors['accent'],
                             activebackground='#E55A2B',
                             relief='flat',
                             cursor='hand2',
                             pady=10)
        close_btn.pack(fill=tk.X, pady=(20, 0))

    def show_help(self):
        help_text = """
SNOWMAN PLANNER VISUALIZER HELP

📁 FILE LOADING:
• Select Problem: Choose a .pddl problem file
• Select Plan: Choose a .txt plan file  
• Load Files: Process and prepare visualization

🎮 ANIMATION CONTROLS:
• Play/Pause: Start or stop animation
• Step Back: Move one frame backward
• Step Forward: Move one frame forward
• Restart: Reset animation to beginning
• Speed Slider: Adjust animation speed

📊 ADDITIONAL FEATURES:
• Metrics: View detailed execution metrics
• Settings: Access additional options
• Modern UI: Enhanced user experience

💡 TIPS:
• Use keyboard shortcuts for faster navigation
• View metrics after animation completion
• Adjust speed for better visualization
        """
        
        # Create modern help window
        help_window = tk.Toplevel(self)
        help_window.title("❓ Help")
        help_window.geometry("500x600")
        help_window.configure(bg=self.colors['background'])
        help_window.resizable(False, False)
        
        # Header
        header = tk.Frame(help_window, bg=self.colors['secondary'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, 
                text="❓ Help & Documentation", 
                font=('Segoe UI', 14, 'bold'),
                fg='white',
                bg=self.colors['secondary']).pack(expand=True)
        
        # Content
        content = tk.Frame(help_window, bg=self.colors['background'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Text widget with scrollbar
        text_frame = tk.Frame(content, bg=self.colors['background'])
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, 
                             font=('Segoe UI', 9),
                             bg=self.colors['surface'],
                             fg=self.colors['text_primary'],
                             wrap=tk.WORD,
                             padx=15, pady=15,
                             relief='flat',
                             borderwidth=1)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        tk.Button(content, 
                 text="✅ Close",
                 command=help_window.destroy,
                 font=('Segoe UI', 10, 'bold'),
                 fg='white',
                 bg=self.colors['accent'],
                 activebackground='#E55A2B',
                 relief='flat',
                 cursor='hand2',
                 pady=10).pack(fill=tk.X, pady=(10, 0))

    def show_about(self):
        about_text = """
🎯 Snowman Planner Visualizer
Version 2.0 - Modern UI Edition

🚀 FEATURES:
• Modern, intuitive interface design
• Real-time animation controls
• Comprehensive metrics analysis
• Enhanced user experience
• Professional visualization tools

💻 DEVELOPED FOR:
• PDDL plan visualization
• Educational purposes
• Research applications
• Planning algorithm analysis

🎨 DESIGN HIGHLIGHTS:
• Material Design inspired UI
• Responsive layout
• Accessible color scheme
• Modern typography
• Smooth animations

📧 SUPPORT:
For technical support and updates,
please contact the development team.

© 2024 Snowman Planner Visualizer
All rights reserved.
        """
        
        # Create modern about window
        about_window = tk.Toplevel(self)
        about_window.title("ℹ️ About")
        about_window.geometry("450x500")
        about_window.configure(bg=self.colors['background'])
        about_window.resizable(False, False)
        
        # Center window
        about_window.transient(self.parent)
        about_window.grab_set()
        
        # Header with gradient effect
        header = tk.Frame(about_window, bg=self.colors['primary'], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, 
                text="ℹ️ About", 
                font=('Segoe UI', 16, 'bold'),
                fg='white',
                bg=self.colors['primary']).pack(expand=True)
        
        # Content
        content = tk.Frame(about_window, bg=self.colors['background'])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # About text
        text_widget = tk.Text(content, 
                             font=('Segoe UI', 9),
                             bg=self.colors['surface'],
                             fg=self.colors['text_primary'],
                             wrap=tk.WORD,
                             padx=15, pady=15,
                             relief='flat',
                             borderwidth=1,
                             height=15)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        text_widget.insert(tk.END, about_text)
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        tk.Button(content, 
                 text="✅ Close",
                 command=about_window.destroy,
                 font=('Segoe UI', 10, 'bold'),
                 fg='white',
                 bg=self.colors['accent'],
                 activebackground='#E55A2B',
                 relief='flat',
                 cursor='hand2',
                 pady=10).pack(fill=tk.X, pady=(10, 0))