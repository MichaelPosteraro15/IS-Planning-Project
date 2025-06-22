import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# File paths (adjust as needed)
PROBLEM_FILE = 'problem-numeric.pddl'
PLAN_FILE = 'plan.txt'

# Helper to convert loc string to grid coords
def parse_loc(loc):
    _, r, c = loc.split('_')
    return int(r) - 1, int(c) - 1

# Parse PDDL problem file
def parse_problem(file_path):
    snow, balls, ball_size = {}, {}, {}
    character = None
    grid_positions = set()
    with open(file_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('(= (location_type'):
            m = re.match(r'\(= \(location_type (\S+)\) (\d+)\)', line)
            if m:
                loc, typ = m.groups()
                coord = parse_loc(loc)
                snow[coord] = (typ == '1')
                grid_positions.add(coord)
        if line.startswith('(ball_at'):
            m = re.match(r'\(ball_at (\S+) (\S+)\)', line)
            if m:
                b, loc = m.groups()
                coord = parse_loc(loc)
                balls[b] = coord
                grid_positions.add(coord)
        for b, size in re.findall(r'\(= \(ball_size (\S+)\) (\d+)\)', line):
            ball_size[b] = int(size)
        if line.startswith('(character_at'):
            m = re.match(r'\(character_at (\S+)\)', line)
            if m:
                loc = m.group(1)
                character = parse_loc(loc)
                grid_positions.add(character)
    for b in balls:
        ball_size.setdefault(b, 0)
    max_r = max(r for r, _ in grid_positions)
    max_c = max(c for _, c in grid_positions)
    return {'snow': snow, 'balls': balls, 'ball_size': ball_size, 'character': character, 'grid_size': max(max_r, max_c) + 1}

# Parse plan file
def parse_plan(file_path):
    steps = []
    pattern = re.compile(r'^\s*(\d+(?:\.\d+)?):\s*\((.+)\)')
    with open(file_path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                t, action = m.groups()
                steps.append((float(t), action.strip()))
    return [step for _, step in sorted(steps)]

# Apply action to state
def apply_action(state, action_str):
    parts = action_str.split()
    new_state = {
        'snow': state['snow'].copy(),
        'balls': state['balls'].copy(),
        'ball_size': state['ball_size'].copy(),
        'character': state['character']
    }
    if parts[0] == 'move_character':
        _, _, to_loc, *_ = parts
        new_state['character'] = parse_loc(to_loc)
    elif parts[0] == 'move_ball':
        _, ball, _, to_loc, *_ = parts
        new_pos = parse_loc(to_loc)
        new_state['balls'][ball] = new_pos
        if state['snow'].get(new_pos, False):
            new_state['ball_size'][ball] += 1
            new_state['snow'][new_pos] = False
    return new_state

# Build state sequence
problem = parse_problem(PROBLEM_FILE)
plan = parse_plan(PLAN_FILE)
states = [problem]
for action in plan:
    states.append(apply_action(states[-1], action))

# Visualization setup
grid_size = problem['grid_size']
interval_ms = 500
is_paused = False

fig, ax = plt.subplots()
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.axis('off')

step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='black')

def draw_person(ax, center, carrying=False, carry_radius=0.15):
    x, y = center
    # Head
    head = patches.Circle((x, y - 0.1), 0.1, facecolor='black', edgecolor='black')
    ax.add_patch(head)
    # Body
    body = patches.Rectangle((x - 0.05, y - 0.1), 0.1, 0.2, facecolor='black', edgecolor='black')
    ax.add_patch(body)
    # Arms
    ax.plot([x - 0.05, x - 0.15], [y, y - 0.05], color='black', linewidth=2)
    ax.plot([x + 0.05, x + 0.15], [y, y - 0.05], color='black', linewidth=2)
    # Legs
    ax.plot([x - 0.05, x - 0.05], [y + 0.1, y + 0.2], color='black', linewidth=2)
    ax.plot([x + 0.05, x + 0.05], [y + 0.1, y + 0.2], color='black', linewidth=2)
    # Carrying indicator
    if carrying:
        ax.add_patch(patches.Circle((x + carry_radius, y - 0.1), carry_radius, facecolor='white', edgecolor='navy', linewidth=2))

def draw_state(state, frame):
    ax.clear()
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    # Background
    for (r, c), is_snow in state['snow'].items():
        face = 'lightcyan' if is_snow else 'lightgreen'
        ax.add_patch(patches.Rectangle((c-0.5, r-0.5), 1, 1, facecolor=face, edgecolor='gray'))
    # Draw stacked balls per cell
    cell_balls = {}
    for b, coord in state['balls'].items():
        cell_balls.setdefault(coord, []).append(b)
    for (r, c), bs in cell_balls.items():
        # sort by size: largest bottom
        bs_sorted = sorted(bs, key=lambda b: state['ball_size'][b], reverse=True)
        n = len(bs_sorted)
        offset_step = 0.2
        for i, b in enumerate(bs_sorted):
            size = state['ball_size'][b]
            radius = 0.2 + 0.1 * size
            # vertical offset
            y_off = (i - (n-1)/2) * offset_step
            circle = patches.Circle((c, r + y_off), radius, fill=True,
                                    edgecolor='navy', facecolor='white', linewidth=2)
            ax.add_patch(circle)
    # Carrying check
    carrying = False
    if frame > 0 and plan[frame - 1].startswith('move_ball'):
        carrying = True
    # Draw person
    cr, cc = state['character']
    draw_person(ax, (cc, cr), carrying=carrying)
    # Step text
    step_text.set_text(f'Step {frame}/{len(states)-1}')

def update(frame):
    draw_state(states[frame], frame)
    return ax.patches + [step_text]

ani = FuncAnimation(fig, update, frames=len(states), interval=interval_ms, blit=True)

def on_key(event):
    global is_paused, interval_ms
    if event.key == ' ':
        if is_paused: ani.event_source.start()
        else: ani.event_source.stop()
        is_paused = not is_paused
    elif event.key in ['+', '=']:
        interval_ms = max(100, interval_ms - 100)
        ani.event_source.interval = interval_ms
    elif event.key == '-':
        interval_ms += 100
        ani.event_source.interval = interval_ms

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
