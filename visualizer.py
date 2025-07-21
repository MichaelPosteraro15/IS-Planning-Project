import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import time

# File paths
PROBLEM_FILE = 'problem-numeric.pddl'
PLAN_FILE = 'plan_numeric.txt'

# Radii for ball sizes
RADIUS = {0: 0.15, 1: 0.25, 2: 0.35}
SHOW_DESCRIPTION = True
SUBSTEPS = 10  # frames per cell movement
PLT_PAUSE = 0.01  # pause duration after each step

# Parsing helpers
def parse_loc(loc):
    parts = loc.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid location format: {loc}")
    _, r, c = parts
    return int(r)-1, int(c)-1

def coord_to_plot(coord, grid_size):
    r, c = coord
    return c, grid_size-1-r

# Parse PDDL problem
def parse_problem(path):
    snow, balls, ball_size = {}, {}, {}
    character = None
    grid_positions = set()
    for line in open(path):
        line = line.strip()
        if line.startswith('(= (location_type'):
            m = re.match(r"\(= \(location_type (\S+)\) (\d+)\)", line)
            if m:
                loc, t = m.groups()
                coord = parse_loc(loc)
                snow[coord] = (t == '1')
                grid_positions.add(coord)
        if line.startswith('(ball_at'):
            match = re.findall(r"\(ball_at (\S+) (\S+)\)", line)
            if match:
                b, loc = match[0]
                balls[b] = parse_loc(loc)
                grid_positions.add(balls[b])
        for b, s in re.findall(r"\(= \(ball_size (\S+)\) (\d+)\)", line):
            ball_size[b] = int(s)
        if line.startswith('(character_at'):
            locs = re.findall(r"\(character_at (\S+)\)", line)
            if locs:
                loc = locs[0]
                character = parse_loc(loc)
                grid_positions.add(character)
    for b in balls:
        ball_size.setdefault(b, 0)
    size = max(max(r for r, _ in grid_positions), max(c for _, c in grid_positions)) + 1
    return {'snow': snow, 'balls': balls, 'ball_size': ball_size, 'character': character, 'grid_size': size}

# Parse plan
def parse_plan(path):
    steps = []
    for line in open(path):
        if ':' in line and '(' in line:
            act = line.split(':', 1)[1].strip()[1:-1]
            if act:
                steps.append(act)
    return steps

# Build frames with interpolation
def build_frames(prob, plan):
    frames = []
    state = {**prob}
    grid = prob['grid_size']

    for i, action in enumerate(plan):
        parts = action.split()
        step_label = f"Step {i + 1}: {action}"

        if parts[0] == 'move_character' and len(parts) >= 3:
            start = parse_loc(parts[1])
            end = parse_loc(parts[2])
            for t in range(SUBSTEPS):
                alpha = t / (SUBSTEPS - 1)
                interp = {
                    'type': 'char_move', 'start': start, 'end': end, 'alpha': alpha,
                    'balls': state['balls'].copy(), 'ball_size': state['ball_size'].copy(),
                    'snow': state['snow'].copy(), 'grid_size': grid,
                    'character': state['character'],
                    'step_text': step_label if t == 0 else None
                }
                frames.append(interp)
            state['character'] = end

        elif parts[0] == 'move_ball' and len(parts) >= 5:
            b, from_cell, mid_cell, to_cell = parts[1], parts[2], parts[3], parts[4]
            start = parse_loc(from_cell)
            end = parse_loc(to_cell)

            # First move character to the ball's start position
            char_start = state['character']
            for t in range(SUBSTEPS):
                alpha = t / (SUBSTEPS - 1)
                interp = {
                    'type': 'char_move', 'start': char_start, 'end': start, 'alpha': alpha,
                    'balls': state['balls'].copy(), 'ball_size': state['ball_size'].copy(),
                    'snow': state['snow'].copy(), 'grid_size': grid,
                    'character': state['character'],
                    'step_text': step_label if t == 0 else None
                }
                frames.append(interp)
            state['character'] = start

            # Then animate the ball move
            for t in range(SUBSTEPS):
                alpha = t / (SUBSTEPS - 1)
                interp = {
                    'type': 'ball_move', 'ball': b, 'start': start, 'end': end, 'alpha': alpha,
                    'balls': state['balls'].copy(), 'ball_size': state['ball_size'].copy(),
                    'snow': state['snow'].copy(), 'character': state['character'], 'grid_size': grid,
                    'step_text': None  # Already printed above
                }
                frames.append(interp)
            state['balls'][b] = end
            if state['snow'].get(end):
                state['ball_size'][b] += 1
                state['snow'][end] = False

    return frames

# Load problem and plan
problem = parse_problem(PROBLEM_FILE)
plan = parse_plan(PLAN_FILE)
frames = build_frames(problem, plan)

# Draw function
def draw(ax, f):
    ax.clear()
    ax.axis('off')
    grid = f['grid_size']
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(-0.5, grid - 0.5)
    for coord, is_snow in f['snow'].items():
        x, y = coord_to_plot(coord, grid)
        color = 'lightcyan' if is_snow else 'lightgreen'
        ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor='gray'))

    for b, pos in f['balls'].items():
        if f['type'] == 'ball_move' and f['ball'] == b:
            sx, sy = coord_to_plot(f['start'], grid)
            ex, ey = coord_to_plot(f['end'], grid)
            x = sx + f['alpha'] * (ex - sx)
            y = sy + f['alpha'] * (ey - sy)
        else:
            x, y = coord_to_plot(pos, grid)
        r = RADIUS.get(f['ball_size'].get(b, 0), 0.15)
        ax.add_patch(patches.Circle((x, y), r, facecolor='white', edgecolor='navy', lw=2))
    if f['type'] == 'char_move':
        sx, sy = coord_to_plot(f['start'], grid)
        ex, ey = coord_to_plot(f['end'], grid)
        cx = sx + f['alpha'] * (ex - sx)
        cy = sy + f['alpha'] * (ey - sy)
    else:
        cx, cy = coord_to_plot(f.get('character', problem['character']), problem['grid_size'])
    ax.add_patch(patches.Circle((cx, cy - 0.1), 0.1, facecolor='black'))
    ax.add_patch(patches.Rectangle((cx - 0.05, cy - 0.1), 0.1, 0.2, facecolor='black'))

    if f.get('step_text'):
        print(f['step_text'])
        ax.text(0, grid + 0.2, f['step_text'], ha='left', va='bottom', fontsize=10, transform=ax.transData)
        plt.pause(PLT_PAUSE)

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
paused = False

# Play/pause button callback
def toggle(event):
    global paused
    paused = not paused

button_ax = plt.axes([0.4, 0.01, 0.2, 0.05])
button = Button(button_ax, 'Play/Pause')
button.on_clicked(toggle)

# Frame update
current_frame = [0]

def update(_):
    if not paused:
        draw(ax, frames[current_frame[0]])
        current_frame[0] = (current_frame[0] + 1) % len(frames)

ani = FuncAnimation(fig, update, frames=len(frames), interval=100)
plt.show()