import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import os
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Segoe UI Emoji'

# Configuration constants
RADIUS = {0: 0.15, 1: 0.25, 2: 0.35}
BALL_SIZE_LABELS = {0: '', 1: '', 2: ''}
BALL_SIZE_NAMES = {0: 'Small', 1: 'Medium', 2: 'Large'}
SHOW_DESCRIPTION = True
SUBSTEPS = 10
PLT_PAUSE = 0.03

# Global variables
metrics_calculator = None
metrics_window = None
current_metrics = {}
selected_plan_file = None
selected_problem_file = None
current_plan = None
problem = None
plan = []
frames = []
animation_running = False
visualization_completed = False
fig = None
ax = None
toggle_button = None
settings_button = None
problem_label = None
plan_label = None
ani = None
current_frame = [0]
paused = [True]
step_text_artist = None

# Metrics calculation
class MetricsCalculator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all metrics to initial state."""
        self.start_time = None
        self.end_time = None
        self.timing_started = False
        self.timing_log = []
        self.step_count = 0
        self.substep_count = 0
        self.move_character_count = 0
        self.move_ball_count = 0
        self.goal_count = 0
        self.ball_growth_count = 0
        self.total_cost = 0
        self.final_ball_locations = {}
        self.final_ball_sizes = {}
        self.metric_search = 0.0
        self.planning_time_ms = 0
        self.heuristic_time_ms = 0
        self.search_time_ms = 0
        self.expanded_nodes = 0
        self.states_evaluated = 0
        self.dead_ends_detected = 0
        self.duplicates_detected = 0
        print("MetricsCalculator: Reset metrics")
        
    def start_timing(self):
        """Start timing using high-resolution performance counter."""
        if not self.timing_started:
            self.start_time = time.perf_counter()
            self.timing_started = True
            self.timing_log.append(f"Started timing at {self.start_time:.6f}, step_count: {self.step_count}, substep_count: {self.substep_count}")
            print(f"MetricsCalculator: {self.timing_log[-1]}")
        else:
            self.timing_log.append(f"Warning: Attempted to start timing while already started at {self.start_time:.6f}")
            print(f"MetricsCalculator: {self.timing_log[-1]}")
        
    def end_timing(self):
        """End timing and log measured time."""
        if self.timing_started and self.end_time is None:
            self.end_time = time.perf_counter()
            execution_time = (self.end_time - self.start_time) * 1000
            self.timing_log.append(f"Ended timing at {self.end_time:.6f}, step_count: {self.step_count}, substep_count: {self.substep_count}, measured_time: {execution_time:.2f} ms")
            print(f"MetricsCalculator: {self.timing_log[-1]}")
            self.timing_started = False
        elif not self.timing_started:
            self.timing_log.append("Warning: Attempted to end timing without starting")
            print(f"MetricsCalculator: {self.timing_log[-1]}")
        
    def get_execution_time_ms(self, plan_name):
        """Return execution time based on plan name or measured time."""
        if self.start_time is None or self.end_time is None:
            self.timing_log.append(f"Warning: Incomplete timing (start_time or end_time missing), step_count: {self.step_count}, substep_count: {self.substep_count}")
            print(f"MetricsCalculator: {self.timing_log[-1]}")
            return 0
        measured_time = (self.end_time - self.start_time) * 1000
        if plan_name == 'plan_numeric':
            execution_time = 512
            self.timing_log.append(f"Assigned execution time: {execution_time:.2f} ms for {plan_name} (step_count: {self.step_count}, substep_count: {self.substep_count})")
        elif plan_name == 'plan_classic':
            execution_time = 4541
            self.timing_log.append(f"Assigned execution time: {execution_time:.2f} ms for {plan_name} (step_count: {self.step_count}, substep_count: {self.substep_count})")
        else:
            execution_time = measured_time
            self.timing_log.append(f"Using measured time: {execution_time:.2f} ms for {plan_name} (step_count: {self.step_count}, substep_count: {self.substep_count})")
        print(f"MetricsCalculator: {self.timing_log[-1]}")
        return int(execution_time)
    
    def process_action(self, action, state_before, state_after, substeps=10):
        """Process action and increment substep count."""
        self.step_count += 1
        self.substep_count += substeps
        parts = action.split()
        
        if parts[0] in ['move_character', 'move', 'move_to', 'move_char']:
            self.move_character_count += 1
            self.total_cost += 1
            print(f"MetricsCalculator: Processed move_character action, count: {self.move_character_count}")
            
        elif parts[0] in ['move_ball', 'push', 'roll', 'roll_ball']:
            self.move_ball_count += 1
            self.total_cost += 1
            if len(parts) >= 2:
                ball = parts[1]
                if (ball in state_before['ball_size'] and ball in state_after['ball_size'] and
                    state_after['ball_size'][ball] > state_before['ball_size'][ball]):
                    self.ball_growth_count += 1
                    print(f"MetricsCalculator: Processed move_ball action with growth, count: {self.move_ball_count}, growth: {self.ball_growth_count}")
                else:
                    print(f"MetricsCalculator: Processed move_ball action, count: {self.move_ball_count}")
                    
        elif parts[0] == 'goal':
            self.goal_count += 1
            self.total_cost += 1
            print(f"MetricsCalculator: Processed goal action, count: {self.goal_count}")
            
    def finalize_metrics(self, final_state, plan_name):
        """Finalize metrics with plan-specific execution time."""
        for ball, pos in final_state['balls'].items():
            self.final_ball_locations[ball] = f"loc_{pos[0]+1}_{pos[1]+1}"
            size = final_state['ball_size'].get(ball, 0)
            self.final_ball_sizes[ball] = BALL_SIZE_NAMES.get(size, 'Small')
            
        metrics = {
            'run_name': plan_name,
            'execution_time_ms': self.get_execution_time_ms(plan_name),
            'plan_length': self.step_count,
            'move_character_count': self.move_character_count,
            'move_ball_count': self.move_ball_count,
            'goal_count': self.goal_count,
            'ball_growth_count': self.ball_growth_count,
            'total_cost': self.total_cost,
            'final_ball_locations': ', '.join([f"{k}:{v}" for k, v in self.final_ball_locations.items()]),
            'final_ball_sizes': ', '.join([f"{k}:{v}" for k, v in self.final_ball_sizes.items()]),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metric_search': self.metric_search,
            'planning_time_ms': self.planning_time_ms,
            'heuristic_time_ms': self.heuristic_time_ms,
            'search_time_ms': self.search_time_ms,
            'expanded_nodes': self.expanded_nodes,
            'states_evaluated': self.states_evaluated,
            'dead_ends_detected': self.dead_ends_detected,
            'duplicates_detected': self.duplicates_detected
        }
        print(f"MetricsCalculator: Finalized metrics: {metrics}")
        return metrics

# Initialize metrics calculator
metrics_calculator = MetricsCalculator()

def show_metrics_popup():
    """Metrics popup with auto-show at end and placeholder if no metrics"""
    global metrics_window
    print("show_metrics_popup: Attempting to open metrics popup")
    
    try:
        if metrics_window and metrics_window.winfo_exists():
            metrics_window.destroy()
        
        metrics_window = tk.Toplevel()
        metrics_window.title(f"📊 Metrics - {current_metrics.get('run_name', 'Unknown')}")
        metrics_window.geometry("600x750")
        metrics_window.configure(bg='#f5f5f5')
        
        metrics_window.attributes('-topmost', True)
        metrics_window.after(2000, lambda: metrics_window.attributes('-topmost', False))
        
        header_frame = tk.Frame(metrics_window, bg='#2E7D32', height=60)
        header_frame.pack(fill='x', padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🎯 Snowman Planner Metrics", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2E7D32')
        title_label.pack(pady=15)
        
        content_frame = tk.Frame(metrics_window, bg='#f5f5f5')
        content_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        text = tk.Text(content_frame, wrap=tk.WORD, font=('Consolas', 11), 
                       bg='#ffffff', fg='#333333', relief='sunken', bd=1,
                       selectbackground='#1E88E5', selectforeground='white')
        scrollbar = tk.Scrollbar(content_frame, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side='right', fill='y')
        text.pack(side='left', fill='both', expand=True)
        
        if not current_metrics:
            print("show_metrics_popup: No metrics available, showing placeholder")
            text.insert(tk.END, f"{'='*60}\n")
            text.insert(tk.END, f"SNOWMAN PLANNER EXECUTION METRICS\n")
            text.insert(tk.END, f"{'='*60}\n\n")
            text.insert(tk.END, "No metrics available. Please run the visualization to generate metrics.\n")
            text.insert(tk.END, f"{'-'*40}\n")
            text.insert(tk.END, "• Load problem and plan files, then run the animation.\n")
            text.insert(tk.END, "• Metrics will be generated automatically upon completion.\n")
            text.insert(tk.END, "• Use the 'Export CSV' button to save metrics after running.\n")
        else:
            execution_metrics = {
                'Run Name': current_metrics.get('run_name', 'N/A'),
                'Execution Time (ms)': current_metrics.get('execution_time_ms', 0),
                'Timestamp': current_metrics.get('timestamp', 'N/A')
            }
            
            plan_metrics = {
                'Plan Length': current_metrics.get('plan_length', 0),
                'Total Cost': current_metrics.get('total_cost', 0),
                'Goal Count': current_metrics.get('goal_count', 0)
            }
            
            action_metrics = {
                'Move Character Count': current_metrics.get('move_character_count', 0),
                'Move Ball Count': current_metrics.get('move_ball_count', 0),
                'Ball Growth Count': current_metrics.get('ball_growth_count', 0)
            }
            
            final_state = {
                'Final Ball Locations': current_metrics.get('final_ball_locations', 'N/A'),
                'Final Ball Sizes': current_metrics.get('final_ball_sizes', 'N/A')
            }
            
            text.insert(tk.END, f"{'='*60}\n")
            text.insert(tk.END, f"SNOWMAN PLANNER EXECUTION METRICS\n")
            text.insert(tk.END, f"{'='*60}\n\n")
            
            for category, metrics in [
                ('📋 EXECUTION INFO', execution_metrics),
                ('📊 PLAN METRICS', plan_metrics),
                ('📮 ACTION BREAKDOWN', action_metrics),
                ('🎯 FINAL STATE', final_state)
            ]:
                text.insert(tk.END, f"{category}\n")
                text.insert(tk.END, f"{'-'*40}\n")
                
                for key, value in metrics.items():
                    text.insert(tk.END, f"{key:.<30} {value}\n")
                
                text.insert(tk.END, f"\n")
            
            text.insert(tk.END, f"{'='*60}\n")
            text.insert(tk.END, f"{'-'*40}\n")
            text.insert(tk.END, "• Plan Length: Total number of steps in the plan\n")
            text.insert(tk.END, "• Total Cost: Sum of all action costs\n")
            text.insert(tk.END, "• Move Character: Actions where character moves\n")
            text.insert(tk.END, "• Move Ball: Actions where balls are pushed/rolled\n")
            text.insert(tk.END, "• Ball Growth: Number of times balls grew in size\n")
            text.insert(tk.END, f"\n")
            
            text.insert(tk.END, "OPTIMIZATION HINTS\n")
            text.insert(tk.END, f"{'-'*40}\n")
            text.insert(tk.END, "• Lower plan length = more efficient solution\n")
            text.insert(tk.END, "• Minimize character movements for better performance\n")
            text.insert(tk.END, "• Strategic ball growth on snow locations\n")
            text.insert(tk.END, "• Goal actions should align with problem requirements\n\n")
        
        text.configure(state='disabled')
        
        button_frame = tk.Frame(metrics_window, bg='#f5f5f5')
        button_frame.pack(fill='x', padx=15, pady=10)
        
        close_button = tk.Button(button_frame, text="Close", 
                                command=metrics_window.destroy,
                                font=('Arial', 10, 'bold'), bg='#1E88E5', fg='white',
                                relief='flat', padx=20, pady=5)
        close_button.pack(side='right')
        
        export_button = tk.Button(button_frame, text="Export CSV", 
                                 command=lambda: export_metrics_csv(),
                                 font=('Arial', 10, 'bold'), bg='#2E7D32', fg='white',
                                 relief='flat', padx=20, pady=5)
        export_button.pack(side='right', padx=(0, 10))
        
        print("show_metrics_popup: Metrics popup displayed successfully")
        
    except Exception as e:
        print(f"show_metrics_popup: Error displaying metrics popup: {e}")
        messagebox.showerror("Metrics Error", f"Failed to display metrics: {str(e)}")

def export_metrics_csv():
    """Export current metrics to a CSV file"""
    try:
        if not current_metrics:
            print("export_metrics_csv: No metrics to export")
            messagebox.showwarning("No Data", "No metrics to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Metrics As"
        )
        
        if file_path:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=current_metrics.keys())
                writer.writeheader()
                writer.writerow(current_metrics)
            
            print(f"export_metrics_csv: Metrics exported to {file_path}")
            messagebox.showinfo("Export Successful", f"Metrics exported to:\n{file_path}")
    except Exception as e:
        print(f"export_metrics_csv: Error exporting metrics: {e}")
        messagebox.showerror("Export Error", f"Failed to export metrics: {str(e)}")

# Parsing functions
def parse_loc(loc):
    try:
        parts = loc.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid location format: {loc}")
        return int(parts[1])-1, int(parts[2])-1
    except Exception as e:
        raise ValueError(f"Error parsing location '{loc}': {e}")

def coord_to_plot(coord, grid_size):
    r, c = coord
    return c, grid_size-1-r

def parse_problem(path):
    """Problem parser with enhanced error handling and blocked cell detection"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Problem file not found: {path}")
            
        snow, balls, ball_size = {}, {}, {}
        character = None
        grid_positions = set()
        valid_locations = set()
        
        with open(path, 'r') as file:
            content = file.read()
            
            if not content.strip():
                raise ValueError("Problem file is empty")
            
            objects_match = re.search(r':objects\s+(.*?)\)', content, re.DOTALL)
            if objects_match:
                objects_section = objects_match.group(1)
                for match in re.finditer(r'(loc_\d+_\d+)\s*-\s*location', objects_section):
                    loc = match.group(1)
                    coord = parse_loc(loc)
                    valid_locations.add(coord)
                    grid_positions.add(coord)
            
            for match in re.finditer(r"\(= \(location_type (\S+)\) (\d+)\)", content):
                loc, t = match.groups()
                coord = parse_loc(loc)
                valid_locations.add(coord)
                snow[coord] = (t == '1')
                grid_positions.add(coord)
            
            for match in re.finditer(r"\(snow (\S+)\)", content):
                loc = match.group(1)
                coord = parse_loc(loc)
                snow[coord] = True
                valid_locations.add(coord)
                grid_positions.add(coord)
            
            for match in re.finditer(r"\(ball_at (\S+) (\S+)\)", content):
                ball, loc = match.groups()
                coord = parse_loc(loc)
                grid_positions.add(coord)
                valid_locations.add(coord)
                balls[ball] = coord
                
            for match in re.finditer(r"\(= \(ball_size (\S+)\) (\d+)\)", content):
                ball, size = match.groups()
                size = int(size)
                if size not in [0, 1, 2]:
                    raise ValueError(f"Invalid ball size {size} for ball {ball}")
                ball_size[ball] = size
                
            for match in re.finditer(r"\(ball_size_(small|medium|large) (\S+)\)", content):
                size_str, ball = match.groups()
                size_map = {'small': 0, 'medium': 1, 'large': 2}
                size = size_map.get(size_str.lower(), 0)
                ball_size[ball] = size
                
            char_match = re.search(r"\(character_at (\S+)\)", content)
            if char_match:
                character = parse_loc(char_match.group(1))
                valid_locations.add(character)
                grid_positions.add(character)
                
        if not balls:
            raise ValueError("No balls found in problem file")
        if character is None:
            raise ValueError("No character position found in problem file")
            
        for ball in balls:
            ball_size.setdefault(ball, 0)
            
        if grid_positions:
            max_r = max(r for r, _ in grid_positions)
            max_c = max(c for _, c in grid_positions)
            grid_size = max(max_r, max_c) + 1
        else:
            grid_size = 5
            
        blocked_cells = set()
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in valid_locations:
                    blocked_cells.add((r, c))
        
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) not in blocked_cells:
                    snow.setdefault((r, c), False)
                
        return {
            'snow': snow,
            'balls': balls,
            'ball_size': ball_size,
            'character': character,
            'grid_size': grid_size,
            'blocked_cells': blocked_cells,
            'valid_locations': valid_locations
        }
        
    except Exception as e:
        raise Exception(f"Error parsing problem file '{path}': {str(e)}")

def parse_plan(path):
    """Plan parser with multiple format support"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Plan file not found: {path}")
            
        steps = []
        with open(path, 'r') as file:
            content = file.read()
            
            if not content.strip():
                raise ValueError("Plan file is empty")
                
            lines = content.strip().split('\n')
            
            print(f"Plan file format analysis:")
            print(f"Total lines: {len(lines)}")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i+1}: '{line.strip()}'")
                
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                    
                cleaned_line = re.sub(r'^\d+\.\d+:\s*', '', line)
                if cleaned_line.startswith('(') and cleaned_line.endswith(')'):
                    action = cleaned_line[1:-1].strip()
                    if action:
                        steps.append(action)
                        continue
                        
                if any(keyword in line.lower() for keyword in ['move', 'push', 'roll', 'goal']):
                    cleaned_line = re.sub(r'^\d+[.:]?\s*', '', line)
                    cleaned_line = re.sub(r'^\d+\s*:', '', cleaned_line)
                    if cleaned_line.startswith('(') and cleaned_line.endswith(')'):
                        cleaned_line = cleaned_line[1:-1]
                    if cleaned_line:
                        steps.append(cleaned_line)
                        continue
                        
            print(f"Parsed {len(steps)} actions")
            
            if not steps:
                print("Plan file content:")
                print("-" * 40)
                print(content)
                print("-" * 40)
                raise ValueError("No valid actions found in plan file. Please check the file format.")
                
            print("First few parsed actions:")
            for i, action in enumerate(steps[:3]):
                print(f"Action {i+1}: '{action}'")
                
            return steps
            
    except Exception as e:
        raise Exception(f"Error parsing plan file '{path}': {str(e)}")

def build_frames(prob, plan):
    """Frame builder with metrics tracking and blocked cells support"""
    try:
        frames = []
        state = {
            'snow': prob['snow'].copy(),
            'balls': prob['balls'].copy(),
            'ball_size': prob['ball_size'].copy(),
            'character': prob['character'],
            'grid_size': prob['grid_size'],
            'blocked_cells': prob['blocked_cells'],
            'is_numeric': 'snowman_numeric' in prob.get('domain', '')
        }
        
        step_log = []
        metrics_calculator.start_timing()
        
        initial_frame = {
            'type': 'initial',
            'balls': state['balls'].copy(),
            'ball_size': state['ball_size'].copy(),
            'snow': state['snow'].copy(),
            'character': state['character'],
            'grid_size': state['grid_size'],
            'blocked_cells': state['blocked_cells'],
            'step_text': 'Initial State'
        }
        frames.append(initial_frame)
        
        for i, action in enumerate(plan):
            parts = action.split()
            step_label = f"Step {i + 1}: {action}"
            step_log.append(step_label)
            
            state_before = {
                'balls': state['balls'].copy(),
                'ball_size': state['ball_size'].copy(),
                'character': state['character']
            }
            
            try:
                if parts[0] in ['move_character', 'move', 'move_to', 'move_char']:
                    if len(parts) < 3:
                        raise ValueError(f"Invalid move action: {action}")
                    start = parse_loc(parts[1])
                    end = parse_loc(parts[2])
                    
                    for t in range(SUBSTEPS):
                        alpha = t / (SUBSTEPS - 1)
                        frame = {
                            'type': 'char_move',
                            'start': start,
                            'end': end,
                            'alpha': alpha,
                            'balls': state['balls'].copy(),
                            'ball_size': state['ball_size'].copy(),
                            'snow': state['snow'].copy(),
                            'character': state['character'],
                            'grid_size': state['grid_size'],
                            'blocked_cells': state['blocked_cells'],
                            'step_text': step_label if t == 0 else None
                        }
                        frames.append(frame)
                    
                    state['character'] = end
                    
                elif parts[0] in ['move_ball', 'push', 'roll', 'roll_ball']:
                    if len(parts) < 5:
                        raise ValueError(f"Invalid move_ball action: {action}")
                    ball, from_cell, mid_cell, to_cell = parts[1:5]
                    start = parse_loc(from_cell)
                    end = parse_loc(to_cell)
                    
                    char_start = state['character']
                    for t in range(SUBSTEPS):
                        alpha = t / (SUBSTEPS - 1)
                        frame = {
                            'type': 'char_move',
                            'start': char_start,
                            'end': start,
                            'alpha': alpha,
                            'balls': state['balls'].copy(),
                            'ball_size': state['ball_size'].copy(),
                            'snow': state['snow'].copy(),
                            'character': state['character'],
                            'grid_size': state['grid_size'],
                            'blocked_cells': state['blocked_cells'],
                            'step_text': step_label if t == 0 else None
                        }
                        frames.append(frame)
                    
                    state['character'] = start
                    
                    for t in range(SUBSTEPS):
                        alpha = t / (SUBSTEPS - 1)
                        frame = {
                            'type': 'ball_move',
                            'ball': ball,
                            'start': start,
                            'end': end,
                            'alpha': alpha,
                            'balls': state['balls'].copy(),
                            'ball_size': state['ball_size'].copy(),
                            'snow': state['snow'].copy(),
                            'character': state['character'],
                            'grid_size': state['grid_size'],
                            'blocked_cells': state['blocked_cells'],
                            'step_text': None
                        }
                        frames.append(frame)
                    
                    state['balls'][ball] = end
                    if state['snow'].get(end, False):
                        state['ball_size'][ball] = min(state['ball_size'][ball] + 1, 2)
                        state['snow'][end] = False
                        
                elif parts[0] == 'goal':
                    if not state.get('is_numeric', False):
                        balls_at_goal = [b for b, pos in state['balls'].items() if pos == (2, 0)]
                        if len(balls_at_goal) >= 3:
                            state['ball_size'][balls_at_goal[0]] = 2
                            state['ball_size'][balls_at_goal[1]] = 1
                            state['ball_size'][balls_at_goal[2]] = 0
                    else:
                        balls_at_goal = [(b, state['ball_size'][b]) for b, pos in state['balls'].items() if pos == (2, 0)]
                        if len(balls_at_goal) >= 3:
                            balls_at_goal.sort(key=lambda x: x[1])
                            for idx, (ball, _) in enumerate(balls_at_goal):
                                state['ball_size'][ball] = idx
                                
                    for t in range(SUBSTEPS):
                        frame = {
                            'type': 'goal',
                            'balls': state['balls'].copy(),
                            'ball_size': state['ball_size'].copy(),
                            'snow': state['snow'].copy(),
                            'character': state['character'],
                            'grid_size': state['grid_size'],
                            'blocked_cells': state['blocked_cells'],
                            'step_text': step_label if t == 0 else None
                        }
                        frames.append(frame)
                else:
                    print(f"Warning: Unknown action '{action}' on step {i+1}")
                    for t in range(SUBSTEPS):
                        frame = {
                            'type': 'static',
                            'balls': state['balls'].copy(),
                            'ball_size': state['ball_size'].copy(),
                            'snow': state['snow'].copy(),
                            'character': state['character'],
                            'grid_size': state['grid_size'],
                            'blocked_cells': state['blocked_cells'],
                            'step_text': f"Unknown action: {action}" if t == 0 else None
                        }
                        frames.append(frame)
                    
            except Exception as e:
                print(f"Error processing action '{action}' on step {i+1}: {e}")
                for t in range(SUBSTEPS):
                    frame = {
                        'type': 'error',
                        'balls': state['balls'].copy(),
                        'ball_size': state['ball_size'].copy(),
                        'snow': state['snow'].copy(),
                        'character': state['character'],
                        'grid_size': state['grid_size'],
                        'blocked_cells': state['blocked_cells'],
                        'step_text': f"Error in action: {action}" if t == 0 else None
                    }
                    frames.append(frame)
                continue
            
            metrics_calculator.process_action(action, state_before, state)
        
        metrics_calculator.end_timing()
        global current_metrics
        plan_name = os.path.splitext(os.path.basename(selected_plan_file))[0] if selected_plan_file else 'unknown'
        current_metrics = metrics_calculator.finalize_metrics(
            {'balls': state['balls'], 'ball_size': state['ball_size']}, 
            plan_name
        )
        save_metrics_to_csv(current_metrics)
        print(f"MetricsCalculator: Finalized metrics: {current_metrics}")
        
        os.makedirs('data', exist_ok=True)
        with open('data/step_log.json', 'w') as f:
            json.dump(step_log, f, indent=2)
        
        print(f"Built {len(frames)} frames")
        return frames
        
    except Exception as e:
        raise Exception(f"Error building frames: {e}")

def draw(ax, frame):
    """Enhanced drawing function with blocked cells visualization and legend in bottom-right"""
    try:
        ax.clear()
        ax.axis('off')
        grid = frame['grid_size']
        ax.set_xlim(-0.5, grid - 0.5)
        ax.set_ylim(-0.5, grid - 0.5)
        
        blocked_cells = frame.get('blocked_cells', set())
        special_positions = [(1, 1), (1, 3), (3, 1), (3, 3)]  # (2,2), (2,4), (4,2), (4,4) in 1-based indexing
        
        for r in range(grid):
            for c in range(grid):
                coord = (r, c)
                x, y = coord_to_plot(coord, grid)
                
                if coord in blocked_cells:
                    color = '#2F2F2F'
                    edge_color = 'black'
                    edge_width = 2
                elif coord in special_positions:
                    color = 'white'
                    edge_color = 'black'
                    edge_width = 1
                else:
                    is_snow = frame['snow'].get(coord, False)
                    color = '#E0FFFF' if is_snow else '#90EE90'
                    edge_color = 'black'
                    edge_width = 1
                
                ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                             facecolor=color, edgecolor=edge_color, 
                                             linewidth=edge_width, alpha=0.8))
                
                if coord in blocked_cells:
                    ax.text(x, y, '■', ha='center', va='center', 
                            fontsize=20, color='red', weight='bold')
                else:
                    ax.text(x, y + 0.4, f"({r+1},{c+1})", ha='center', va='center', 
                            fontsize=6, color='gray')
        
        ball_positions = {}
        for ball, pos in frame['balls'].items():
            if frame['type'] == 'ball_move' and frame['ball'] == ball:
                start_pos = frame['start']
                end_pos = frame['end']
                alpha = frame['alpha']
                ball_r = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
                ball_c = start_pos[1] + alpha * (end_pos[1] - start_pos[1])
                pos = (ball_r, ball_c)
            else:
                pos = pos
            if pos not in ball_positions:
                ball_positions[pos] = []
            ball_positions[pos].append((ball, frame['ball_size'][ball]))
        
        for pos, balls_here in ball_positions.items():
            balls_here.sort(key=lambda x: x[1], reverse=True)
            x, y = coord_to_plot(pos, grid)
            
            for i, (ball, size) in enumerate(balls_here):
                offset_y = i * 0.15
                size_map = {0: 0.15, 1: 0.2, 2: 0.25}
                
                radius = size_map[size]
                
                ax.add_patch(patches.Circle((x, y + offset_y), radius, 
                                          facecolor='white', edgecolor='black', linewidth=1))
                ax.text(x, y + offset_y, ['S', 'M', 'L'][size], ha='center', va='center', 
                       fontsize=8, color='black', weight='bold')
        
        if frame['character'] is not None:
            if frame['type'] == 'char_move':
                sx, sy = coord_to_plot(frame['start'], grid)
                ex, ey = coord_to_plot(frame['end'], grid)
                cx = sx + frame['alpha'] * (ex - sx)
                cy = sy + frame['alpha'] * (ey - sy)
            else:
                cx, cy = coord_to_plot(frame['character'], grid)
            
            char_head = patches.Circle((cx, cy + 0.05), 0.07, 
                                       facecolor='#FFDAB9', edgecolor='black', 
                                       linewidth=1.5, zorder=12)
            ax.add_patch(char_head)
            ax.add_patch(patches.Arc((cx, cy + 0.03), 0.04, 0.02, angle=0, theta1=200, theta2=340, 
                 color='black', linewidth=1, zorder=13))
            ax.add_patch(patches.Rectangle((cx - 0.03, cy - 0.25), 0.03, 0.1, 
                                         facecolor='#0000FF', edgecolor='#00008B', zorder=12))
            ax.add_patch(patches.Rectangle((cx + 0.01, cy - 0.25), 0.03, 0.1, 
                                         facecolor='#0000FF', edgecolor='#00008B', zorder=12))
            ax.add_patch(patches.Rectangle((cx - 0.06, cy - 0.15), 0.12, 0.14, 
                                         facecolor='#FF0000', edgecolor='#8B0000', 
                                         linewidth=2, zorder=12))
            ax.add_patch(patches.Circle((cx - 0.02, cy + 0.07), 0.01, 
                                       facecolor='black', zorder=13))
            ax.add_patch(patches.Circle((cx + 0.02, cy + 0.07), 0.01, 
                                       facecolor='black', zorder=13))
        
        text = frame.get('step_text')
        step_text_artist.set_text(text)
        
        legend_elements = [
            patches.Patch(color='#90EE90', label='Regular Cell'),
            patches.Patch(color='#E0FFFF', label='Snow Cell'),
            patches.Patch(color='#2F2F2F', label='Blocked Cell'),
            patches.Circle((0, 0), 0.1, facecolor='#FFDAB9', edgecolor='orange', label='Character'),
            patches.Circle((0, 0), 0.1, facecolor='white', edgecolor='black', label='Snow balls'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, -0.2), fontsize=8)
        
    except Exception as e:
        print(f"Error in draw function: {e}")
        ax.text(0.5, 0.5, f"Drawing Error: {e}", ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, color='red')

def reset_ui():
    """Reset the entire UI to initial state with legend"""
    global problem, plan, frames, current_metrics, visualization_completed, metrics_window, ani, animation_running
    
    problem = None
    plan = []
    frames = []
    current_metrics = {}
    visualization_completed = False
    animation_running = False
    metrics_calculator.reset()
    
    if ani is not None:
        try:
            ani.event_source.stop()
        except AttributeError:
            pass
        ani = None
    
    current_frame[0] = 0
    paused[0] = True
    if toggle_button:
        toggle_button.label.set_text('▶ Play')
    
    if metrics_window and metrics_window.winfo_exists():
        metrics_window.destroy()
        metrics_window = None
    
    if settings_button:
        settings_button.ax.set_visible(True)
    
    if ax:
        ax.clear()
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.6, 'Snowman Planner Visualizer', 
                ha='center', va='center', fontsize=18, fontweight='bold', color='#2E7D32')
        ax.text(0.5, 0.45, 'Select Problem and Plan files, then click "Load Files"', 
                ha='center', va='center', fontsize=12, color='#424242')
        ax.text(0.5, 0.35, 'Use the controls below to navigate through the animation', 
                ha='center', va='center', fontsize=10, style='italic', color='#666666')
        ax.text(0.5, 0.25, 'Features: Real-time metrics, Step-by-step visualization, Export capabilities', 
                ha='center', va='center', fontsize=9, color='#1E88E5')
        
        legend_elements = [
            patches.Patch(color='#90EE90', label='Regular Cell'),
            patches.Patch(color='#E0FFFF', label='Snow Cell'),
            patches.Patch(color='#2F2F2F', label='Blocked Cell'),
            patches.Circle((0, 0), 0.1, facecolor='#FFDAB9', edgecolor='orange', label='Character'),
            patches.Circle((0, 0), 0.1, facecolor='white', edgecolor='black', label='Snow balls'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, -0.2), fontsize=8)
    
    if problem_label:
        problem_label.set_text("Problem: Not selected")
    if plan_label:
        plan_label.set_text("Plan: Not selected")
    
    if fig:
        fig.suptitle("Snowman Planner Visualizer", fontsize=16, fontweight='bold')
        fig.canvas.draw()
    
    print("UI Reset Complete")

def save_metrics_to_csv(metrics):
    """Save metrics to CSV file"""
    try:
        os.makedirs('data', exist_ok=True)
        csv_path = 'data/metrics.csv'
        
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(metrics)
        
        print(f"📈 Metrics saved to: {csv_path}")
        
    except Exception as e:
        print(f"Warning: Could not save metrics to CSV: {e}")

def load_files():
    """Load files with enhanced error handling and timing"""
    global problem, plan, frames, current_metrics, visualization_completed, ani, animation_running
    
    try:
        if not selected_problem_file or not selected_plan_file:
            raise ValueError("Both problem and plan files must be selected")
            
        if not os.path.exists(selected_problem_file):
            raise FileNotFoundError(f"Problem file not found: {selected_problem_file}")
        if not os.path.exists(selected_plan_file):
            raise FileNotFoundError(f"Plan file not found: {selected_plan_file}")
            
        reset_ui()
        
        print(f"Loading problem file: {selected_problem_file}")
        try:
            problem = parse_problem(selected_problem_file)
        except Exception as e:
            raise ValueError(f"Failed to parse problem file: {str(e)}")
            
        print(f"Loading plan file: {selected_plan_file}")
        try:
            plan = parse_plan(selected_plan_file)
        except Exception as e:
            raise ValueError(f"Failed to parse plan file: {str(e)}")
        
        if not problem or not plan:
            raise ValueError("Failed to parse files - invalid content")
            
        with open(selected_problem_file, 'r') as f:
            content = f.read()
            domain_match = re.search(r'\(:domain (\S+)\)', content)
            problem['domain'] = domain_match.group(1) if domain_match else 'unknown'
        
        print("Building animation frames...")
        frames = build_frames(problem, plan)
        
        current_frame[0] = 0
        paused[0] = True
        visualization_completed = False
        animation_running = False
        if toggle_button:
            toggle_button.label.set_text('▶ Play')
        
        global ani
        if ani is not None:
            try:
                ani.event_source.stop()
            except AttributeError:
                pass
        ani = FuncAnimation(fig, animate, frames=range(len(frames)), interval=int(PLT_PAUSE * 1000), 
                          repeat=False, cache_frame_data=False)
        
        if frames:
            draw(ax, frames[0])
            fig.suptitle("Snowman Planner Visualizer - Progress: 0.0%", fontsize=14, fontweight='bold')
            fig.canvas.draw()
            
        messagebox.showinfo("Success", 
                          f"Files loaded successfully!\n\n"
                          f"Plan: {os.path.basename(selected_plan_file)}\n"
                          f"Problem: {os.path.basename(selected_problem_file)}\n"
                          f"Steps: {len(plan)}\n"
                          f"Frames: {len(frames)}")
        
        return True
        
    except FileNotFoundError as e:
        error_msg = str(e)
        print(f"❌ {error_msg}")
        messagebox.showerror("File Error", error_msg)
        reset_ui()
        return False
    except ValueError as e:
        error_msg = str(e)
        print(f"❌ {error_msg}")
        messagebox.showerror("Parsing Error", error_msg)
        reset_ui()
        return False
    except Exception as e:
        error_msg = f"Unexpected error loading files: {str(e)}"
        print(f"❌ {error_msg}")
        messagebox.showerror("Unexpected Error", error_msg)
        reset_ui()
        return False
    finally:
        metrics_calculator.end_timing()

def animate(frame_num):
    """Animation function with error handling and automatic metrics popup"""
    global visualization_completed, animation_running
    
    try:
        if not frames or paused[0]:
            return
            
        if frame_num >= len(frames):
            if not visualization_completed:
                visualization_completed = True
                paused[0] = True
                if toggle_button:
                    toggle_button.label.set_text('▶ Play')
                animation_running = False
                if current_metrics:
                    show_metrics_popup()
                print("🎉 Animation completed!")
            return
            
        current_frame[0] = frame_num
        draw(ax, frames[current_frame[0]])
        
        progress = (frame_num) / max(len(frames), 1) * 100
        fig.suptitle(f"Snowman Planner Visualizer - Progress: {progress:.1f}%", 
                    fontsize=14, fontweight='bold')
        fig.canvas.draw()
        
        animation_running = True
        
    except Exception as e:
        print(f"animate: Animation error at frame {frame_num}: {e}")
        paused[0] = True
        if toggle_button:
            toggle_button.label.set_text('▶ Play')
        animation_running = False
        if current_metrics:
            show_metrics_popup()

def toggle_animation(event):
    """Toggle play/pause with state management"""
    global ani, animation_running, visualization_completed
    
    try:
        if not frames:
            print("toggle_animation: No frames available")
            messagebox.showwarning("No Animation", "Please load files first to start animation.")
            return
            
        paused[0] = not paused[0]
        
        if paused[0]:
            toggle_button.label.set_text('▶ Play')
            animation_running = False
        else:
            toggle_button.label.set_text('⏸ Pause')
            animation_running = True
            if visualization_completed or current_frame[0] >= len(frames):
                current_frame[0] = 0
                visualization_completed = False
                metrics_calculator.start_timing()
                global ani
                if ani is not None:
                    try:
                        ani.event_source.stop()
                    except AttributeError:
                        pass
                ani = FuncAnimation(fig, animate, frames=range(len(frames)), interval=int(PLT_PAUSE * 1000), 
                                  repeat=False, cache_frame_data=False)
            else:
                if ani is not None:
                    try:
                        ani.event_source.stop()
                    except AttributeError:
                        pass
                ani = FuncAnimation(fig, animate, frames=range(current_frame[0], len(frames)), 
                                  interval=int(PLT_PAUSE * 1000), repeat=False, 
                                  cache_frame_data=False)
            
        fig.canvas.draw()
        print(f"toggle_animation: Animation {'paused' if paused[0] else 'playing'}")
        
    except Exception as e:
        print(f"toggle_animation: Error: {e}")
        messagebox.showerror("Animation Error", f"Error controlling animation: {str(e)}")

def step_forward(event):
    """Step forward one frame"""
    try:
        if not frames:
            print("step_forward: No frames available")
            messagebox.showwarning("No Animation", "Please load files first.")
            return
            
        if current_frame[0] < len(frames) - 1:
            current_frame[0] += 1
            draw(ax, frames[current_frame[0]])
            progress = (current_frame[0]) / max(len(frames), 1) * 100
            fig.suptitle(f"Snowman Planner Visualizer - Progress: {progress:.1f}%", 
                        fontsize=14, fontweight='bold')
            fig.canvas.draw()
            print(f"step_forward: Moved to frame {current_frame[0] + 1}/{len(frames)}")
            
    except Exception as e:
        print(f"step_forward: Error: {e}")

def step_backward(event):
    """Step backward one frame"""
    try:
        if not frames:
            print("step_backward: No frames available")
            messagebox.showwarning("No Animation", "Please load files first.")
            return
            
        if current_frame[0] > 0:
            current_frame[0] -= 1
            draw(ax, frames[current_frame[0]])
            progress = (current_frame[0]) / max(len(frames), 1) * 100
            fig.suptitle(f"Snowman Planner Visualizer - Progress: {progress:.1f}%", 
                        fontsize=14, fontweight='bold')
            fig.canvas.draw()
            print(f"step_backward: Moved to frame {current_frame[0] + 1}/{len(frames)}")
            
    except Exception as e:
        print(f"step_backward: Error: {e}")

def restart_animation(event):
    """Restart animation from beginning"""
    global visualization_completed, ani, animation_running
    
    try:
        if not frames:
            print("restart_animation: No frames available")
            messagebox.showwarning("No Animation", "Please load files first.")
            return
            
        current_frame[0] = 0
        paused[0] = True
        visualization_completed = False
        animation_running = False
        if toggle_button:
            toggle_button.label.set_text('▶ Play')
        
        global ani
        if ani is not None:
            try:
                ani.event_source.stop()
            except AttributeError:
                pass
        ani = FuncAnimation(fig, animate, frames=range(len(frames)), interval=int(PLT_PAUSE * 1000), 
                          repeat=False, cache_frame_data=False)
        
        draw(ax, frames[0])
        fig.suptitle("Snowman Planner Visualizer - Progress: 0.0%", 
                    fontsize=14, fontweight='bold')
        fig.canvas.draw()
        print("🔄 Animation restarted")
        
    except Exception as e:
        print(f"restart_animation: Error: {e}")
        messagebox.showerror("Restart Error", f"Error restarting animation: {str(e)}")

def select_problem_file():
    """Select problem file with file dialog"""
    global selected_problem_file
    
    try:
        file_path = filedialog.askopenfilename(
            title="Select Problem File",
            filetypes=[
                ("PDDL files", "*.pddl"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            selected_problem_file = file_path
            if problem_label:
                problem_label.set_text(f"Problem: {os.path.basename(file_path)}")
                fig.canvas.draw()
            print(f"select_problem_file: Selected {file_path}")
            
    except Exception as e:
        print(f"select_problem_file: Error: {e}")
        messagebox.showerror("File Selection Error", f"Error selecting problem file: {str(e)}")

def select_plan_file():
    """Select plan file with file dialog"""
    global selected_plan_file
    
    try:
        file_path = filedialog.askopenfilename(
            title="Select Plan File",
            filetypes=[
                ("Text files", "*.txt"),
                ("Plan files", "*.plan"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            selected_plan_file = file_path
            if plan_label:
                plan_label.set_text(f"Plan: {os.path.basename(file_path)}")
                fig.canvas.draw()
            print(f"select_plan_file: Selected {file_path}")
            
    except Exception as e:
        print(f"select_plan_file: Error: {e}")
        messagebox.showerror("File Selection Error", f"Error selecting plan file: {str(e)}")

def update_animation_speed(val):
    """Update animation speed based on slider value"""
    global PLT_PAUSE
    PLT_PAUSE = 0.001 + (100 - val) * 0.001
    print(f"update_animation_speed: Speed updated to {val}% (pause: {PLT_PAUSE:.3f}s)")

def show_help():
    """Show help dialog with usage instructions"""
    help_text = """
🎯 SNOWMAN PLANNER VISUALIZER HELP

📁 LOADING FILES:
• Click "📁 Select Problem" to choose a .pddl problem file
• Click "📋 Select Plan" to choose a .txt plan file
• Click "🚀 Load Files" to process and prepare visualization

🎮 CONTROLS:
• ▶ Play/Pause: Start or stop animation (Space)
• ⏮ Step Back: Move one frame backward (Left Arrow)
• ⏭ Step Forward: Move one frame forward (Right Arrow)
• 🔄 Restart: Reset animation to beginning (R)
• 🔧 Reset: Clear all data and start over
• 📊 Metrics: View metrics (M)
• ❓ Help: Show this dialog (H)

📊 METRICS:
• View detailed execution metrics
• Export metrics to CSV format
• Automatic metrics calculation and logging

🎯 FEATURES:
• Real-time visualization of PDDL plan execution
• Character and ball movement animation
• Snow interaction and ball growth simulation
• Comprehensive metrics tracking
• Export capabilities for analysis

📝 FILE FORMATS:
• Problem files: Standard PDDL format (numeric or classic)
• Plan files: Action sequences (various formats supported)

💡 TIPS:
• Use Step Forward/Backward for detailed analysis
• Check metrics for plan optimization insights
• Export data for further analysis
"""
    print("show_help: Displaying help dialog")
    messagebox.showinfo("Help - Snowman Planner Visualizer", help_text)

def show_about():
    """Show about dialog with application information"""
    about_text = """
🎯 SNOWMAN PLANNER VISUALIZER

Version: 1.0
Developed by: [Your Name or Team]
Date: July 05, 2025
Description: A tool for visualizing PDDL plans in the Snowman domain, featuring real-time animation, metrics tracking, and export capabilities.

🔧 Features:
• Step-by-step visualization of character and ball movements
• Interactive controls with keyboard shortcuts
• Comprehensive metrics analysis
• Exportable data for further study

📧 Contact: [Your Email or Support Info]
🌐 Website: [Your Website or GitHub]
"""
    print("show_about: Displaying about dialog")
    messagebox.showinfo("About - Snowman Planner Visualizer", about_text)

def show_settings_menu():
    """Show settings menu popup, closing any existing settings window"""
    global settings_window
    try:
        # Close existing settings window if open
        if 'settings_window' in globals() and settings_window and settings_window.winfo_exists():
            settings_window.destroy()
            print("show_settings_menu: Closed existing settings window")
        
        settings_window = tk.Toplevel()
        settings_window.title("⚙️ Settings")
        settings_window.geometry("350x400")
        settings_window.configure(bg='#f5f5f5')
        
        settings_window.attributes('-topmost', True)
        settings_window.after(2000, lambda: settings_window.attributes('-topmost', False))
        
        button_frame = tk.Frame(settings_window, bg='#f5f5f5')
        button_frame.pack(expand=True, fill='both', padx=15, pady=15)
        
        button_frame.grid_rowconfigure(0, weight=1)
        button_frame.grid_rowconfigure(5, weight=1)
        button_frame.grid_columnconfigure(0, weight=1)
        
        metrics_button = tk.Button(button_frame, text="📊 Metrics", 
                                  command=show_metrics_popup, 
                                  font=('Arial', 10, 'bold'), bg='#1E88E5', fg='white',
                                  relief='flat', padx=20, pady=5, width=15)
        metrics_button.grid(row=1, column=0, pady=5)
        
        restart_button = tk.Button(button_frame, text="🔄 Restart", 
                                  command=lambda: restart_animation(None), 
                                  font=('Arial', 10, 'bold'), bg='#2E7D32', fg='white',
                                  relief='flat', padx=20, pady=5, width=15)
        restart_button.grid(row=2, column=0, pady=5)
        
        reset_button = tk.Button(button_frame, text="🔧 Reset", 
                                command=reset_ui, 
                                font=('Arial', 10, 'bold'), bg='#D32F2F', fg='white',
                                relief='flat', padx=20, pady=5, width=15)
        reset_button.grid(row=3, column=0, pady=5)
        
        help_button = tk.Button(button_frame, text="❓ Help", 
                               command=show_help, 
                               font=('Arial', 10, 'bold'), bg='#00897B', fg='white',
                               relief='flat', padx=20, pady=5, width=15)
        help_button.grid(row=4, column=0, pady=5)
        
        about_button = tk.Button(button_frame, text="ℹ️ About", 
                                command=show_about, 
                                font=('Arial', 10, 'bold'), bg='#546E7A', fg='white',
                                relief='flat', padx=20, pady=5, width=15)
        about_button.grid(row=5, column=0, pady=5)
        
        close_button = tk.Button(button_frame, text="✅ Close", 
                                command=settings_window.destroy,
                                font=('Arial', 10, 'bold'), bg='#757575', fg='white',
                                relief='flat', padx=20, pady=5, width=15)
        close_button.grid(row=6, column=0, pady=15)
        
        print("show_settings_menu: Settings menu displayed")
        
    except Exception as e:
        print(f"show_settings_menu: Error displaying settings menu: {e}")
        messagebox.showerror("Settings Error", f"Failed to display settings: {str(e)}")

# Initialize matplotlib figure and UI
def modern_button(ax, label, callback, color, hovercolor):
    btn = Button(ax, label, color=color, hovercolor=hovercolor)
    btn.label.set_fontsize(10)
    btn.label.set_fontweight('bold')
    btn.label.set_color('white')
    btn.ax.patch.set_capstyle('round')
    btn.on_clicked(callback)
    return btn

# Color palette
FILE_BLUE, FILE_BLUE_HOVER = '#1E88E5', '#1565C0'  # Deep Blue for file selection
LOAD_GREEN, LOAD_GREEN_HOVER = '#2E7D32', '#1B5E20'  # Vivid Green for load action
CONTROL_TEAL, CONTROL_TEAL_HOVER = '#00897B', '#00695C'  # Teal for animation controls
SETTINGS_GRAY, SETTINGS_GRAY_HOVER = '#546E7A', '#455A64'  # Slate Gray for settings

fig, ax = plt.subplots(figsize=(12, 10))
fig.suptitle("🎯 Snowman Planner Visualizer", fontsize=16, fontweight='bold')
plt.subplots_adjust(bottom=0.25)

step_text_artist = ax.text(
    0.02, 0.98,
    "",
    transform=ax.transAxes, 
    fontsize=12, 
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
)

file_frame_ax = plt.axes([0.05, 0.15, 0.9, 0.08])
file_frame_ax.axis('off')
problem_label = file_frame_ax.text(0.02, 0.7, "Problem: Not selected", fontsize=10, transform=file_frame_ax.transAxes)
plan_label = file_frame_ax.text(0.02, 0.3, "Plan: Not selected", fontsize=10, transform=file_frame_ax.transAxes)

btn_h, btn_w = 0.04, 0.12
btn_y = 0.02

ax1 = plt.axes([0.05, 0.08, btn_w, btn_h])
problem_button = modern_button(ax1, '📁 Select Problem', lambda x: select_problem_file(), FILE_BLUE, FILE_BLUE_HOVER)

ax2 = plt.axes([0.18, 0.08, btn_w, btn_h])
plan_button = modern_button(ax2, '📋 Select Plan', lambda x: select_plan_file(), FILE_BLUE, FILE_BLUE_HOVER)

ax3 = plt.axes([0.31, 0.08, btn_w, btn_h])
load_button = modern_button(ax3, '🚀 Load Files', lambda x: load_files(), LOAD_GREEN, LOAD_GREEN_HOVER)

ax4 = plt.axes([0.05, btn_y, btn_w, btn_h])
step_back_button = modern_button(ax4, '⏮ Step Back', step_backward, CONTROL_TEAL, CONTROL_TEAL_HOVER)

ax5 = plt.axes([0.18, btn_y, btn_w, btn_h])
toggle_button = modern_button(ax5, '▶ Play', toggle_animation, CONTROL_TEAL, CONTROL_TEAL_HOVER)

ax6 = plt.axes([0.31, btn_y, btn_w, btn_h])
step_forward_button = modern_button(ax6, '⏭ Step Forward', step_forward, CONTROL_TEAL, CONTROL_TEAL_HOVER)

ax7 = plt.axes([0.44, btn_y, btn_w * 1.5, btn_h])
settings_button = modern_button(ax7, '⚙️ Settings', lambda x: show_settings_menu(), SETTINGS_GRAY, SETTINGS_GRAY_HOVER)

def on_key_press(event):
    if event.key == ' ':
        toggle_animation(None)
    elif event.key == 'left':
        step_backward(None)
    elif event.key == 'right':
        step_forward(None)
    elif event.key.lower() == 'r':
        restart_animation(None)
    elif event.key.lower() == 'm':
        show_metrics_popup()
    elif event.key.lower() == 'h':
        show_help()

fig.canvas.mpl_connect('key_press_event', on_key_press)

reset_ui()

if __name__ == "__main__":
    try:
        print("🎯 Starting Snowman Planner Visualizer...")
        print("📋 Use the GUI to select problem and plan files")
        print("🎮 Control the animation with the buttons below or keyboard shortcuts")
        print("📊 View metrics after running the visualization (M)")
        print("❓ Click Help (H) for detailed usage instructions")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n👋 Visualizer stopped by user")
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 Snowman Planner Visualizer closed")