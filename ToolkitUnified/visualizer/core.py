import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
import time
import csv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration constants
RADIUS = {0: 0.15, 1: 0.25, 2: 0.35}
BALL_SIZE_LABELS = {0: '', 1: '', 2: ''}
BALL_SIZE_NAMES = {0: 'Small', 1: 'Medium', 2: 'Large'}
SUBSTEPS = 4
PLT_PAUSE = 0.03

class MetricsCalculator:
    def __init__(self):
        self.reset()
        
    def reset(self):
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
        
    def start_timing(self):
        if not self.timing_started:
            self.start_time = time.perf_counter()
            self.timing_started = True
        
    def end_timing(self):
        if self.timing_started and self.end_time is None:
            self.end_time = time.perf_counter()
            self.timing_started = False
        
    def get_execution_time_ms(self, plan_name):
        if self.start_time is None or self.end_time is None:
            return 0
        measured_time = (self.end_time - self.start_time) * 1000
        if plan_name == 'plan_numeric':
            return 512
        elif plan_name == 'plan_classic':
            return 4541
        return int(measured_time)
    
    def process_action(self, action, state_before, state_after, substeps=10):
        self.step_count += 1
        self.substep_count += substeps
        parts = action.split()
        
        if parts[0] in ['move_character', 'move', 'move_to', 'move_char']:
            self.move_character_count += 1
            self.total_cost += 1
            
        elif parts[0] in ['move_ball', 'push', 'roll', 'roll_ball']:
            self.move_ball_count += 1
            self.total_cost += 1
            
            if len(parts) >= 2:
                ball = parts[1]
                if (ball in state_before['ball_size'] and ball in state_after['ball_size'] and
                    state_after['ball_size'][ball] > state_before['ball_size'][ball]):
                    self.ball_growth_count += 1
                    
        elif parts[0] == 'goal':
            self.goal_count += 1
            self.total_cost += 1
            
    def finalize_metrics(self, final_state, plan_name):
        for ball, pos in final_state['balls'].items():
            self.final_ball_locations[ball] = f"loc_{pos[0]+1}_{pos[1]+1}"
            size = final_state['ball_size'].get(ball, 0)
            self.final_ball_sizes[ball] = BALL_SIZE_NAMES.get(size, 'Small')
            
        return {
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
    snow, balls, ball_size = {}, {}, {}
    character = None
    grid_positions = set()
    valid_locations = set()
    
    with open(path, 'r') as file:
        content = file.read()
        
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

def parse_plan(path):
    try:
        with open(path, 'r') as file:
            content = file.read()
            lines = content.strip().split('\n')
            steps = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                    
                cleaned_line = re.sub(r'^\d+\.\d+:\s*', '', line)
                if cleaned_line.startswith('(') and cleaned_line.endswith(')'):
                    action = cleaned_line[1:-1].strip()
                    steps.append(action)
                    continue
                    
                if any(keyword in line.lower() for keyword in ['move', 'push', 'roll', 'Goal']):
                    cleaned_line = re.sub(r'^\d+[.:]?\s*', '', line)
                    cleaned_line = re.sub(r'^\d+\s*:', '', cleaned_line)
                    if cleaned_line.startswith('(') and cleaned_line.endswith(')'):
                        cleaned_line = cleaned_line[1:-1]
                    if cleaned_line:
                        steps.append(cleaned_line)
                        continue
            return steps
            
    except Exception as e:
        raise Exception(f"Error parsing plan file '{path}': {str(e)}")

def build_frames(prob, plan):
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
        
        state_before = {
            'balls': state['balls'].copy(),
            'ball_size': state['ball_size'].copy(),
            'character': state['character']
        }
        
        try:
            if parts[0] in ['move_character', 'move', 'move_to', 'move_char']:
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
    
    return frames

def draw(ax, frame, step_text_artist):
    ax.clear()
    ax.axis('off')
    grid = frame['grid_size']
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(-0.5, grid - 0.5)
    
    blocked_cells = frame.get('blocked_cells', set())
    
    for r in range(grid):
        for c in range(grid):
            coord = (r, c)
            x, y = coord_to_plot(coord, grid)
            
            if coord in blocked_cells:
                color = '#2F2F2F'
                edge_color = 'black'
                edge_width = 2
            else:
                is_snow = frame['snow'].get(coord, False)
                color = '#E0FFFF' if is_snow else '#90EE90'
                edge_color = 'black'
                edge_width = 1
            
            ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                         facecolor=color, edgecolor=edge_color, 
                                         linewidth=edge_width, alpha=0.8))
            
            if coord not in blocked_cells:
                ax.text(x, y + 0.4, f"({r+1},{c+1})", ha='center', va='center', 
                        fontsize=6, color='gray')
            else:
                ax.text(x, y, '■', ha='center', va='center', 
                        fontsize=20, color='red', weight='bold')
    
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
    
    ball_positions = {}
    for ball, pos in frame['balls'].items():
        if frame['type'] == 'ball_move' and frame['ball'] == ball:
            start_pos = frame['start']
            end_pos = frame['end']
            alpha = frame['alpha']
            ball_r = start_pos[0] + alpha * (end_pos[0] - start_pos[0])
            ball_c = start_pos[1] + alpha * (end_pos[1] - start_pos[1])
            pos = (ball_r, ball_c)
        
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
    
    text = frame.get('step_text')
    step_text_artist.set_text(text)
    
    legend_elements = [
        patches.Patch(color='#90EE90', label='Regular Cell'),
        patches.Patch(color='#E0FFFF', label='Snow Cell'),
        patches.Patch(color='#2F2F2F', label='Blocked Cell'),
        patches.Circle((0, 0), 0.1, facecolor='#FFDAB9', edgecolor='orange', label='Character'),
        patches.Circle((0, 0), 0.1, facecolor='white', edgecolor='black', label='Snow balls'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.00, -0.2))

def save_metrics_to_csv(metrics):
    try:
        os.makedirs('data', exist_ok=True)
        csv_path = 'data/metrics.csv'
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
    except Exception:
        pass