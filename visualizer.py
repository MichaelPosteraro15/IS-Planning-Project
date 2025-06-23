import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# File paths (adjust as needed)
PROBLEM_FILE = 'problem-numeric.pddl'
PLAN_FILE = 'plan_numeric.txt'

SMALL_RADIUS  = 0.15
MEDIUM_RADIUS = 0.25
LARGE_RADIUS  = 0.35
SHOW_DESCRIPTION = True  # Toggle to show action description text

def parse_loc(loc):
    _, r, c = loc.split('_')
    return (int(r) - 1, int(c) - 1)

def coord_to_plot(coord, grid_size):
    r, c = coord
    # flip y-axis for plotting
    return c, grid_size - 1 - r

# Parsing functions
def parse_problem(file_path):
    snow, balls, ball_size = {}, {}, {}
    character = None
    grid_positions = set()
    with open(file_path) as f:
        for line in f:
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
                    b, l = m.groups()
                    coord = parse_loc(l)
                    balls[b] = coord
                    grid_positions.add(coord)
            for b, size in re.findall(r'\(= \(ball_size (\S+)\) (\d+)\)', line):
                ball_size[b] = int(size)
            if line.startswith('(character_at'):
                m = re.match(r'\(character_at (\S+)\)', line)
                if m:
                    coord = parse_loc(m.group(1))
                    character = coord
                    grid_positions.add(coord)
    for b in balls:
        ball_size.setdefault(b, 0)
    size = max(max(r for r, _ in grid_positions),
               max(c for _, c in grid_positions)) + 1
    return {'snow': snow, 'balls': balls, 'ball_size': ball_size,
            'character': character, 'grid_size': size}

def parse_plan(file_path):
    steps = []
    pattern = re.compile(r'^\s*\d+(?:\.\d+)?:\s*\((.+)\)')
    with open(file_path) as f:
        for line in f:
            m = pattern.match(line)
            if m:
                steps.append(m.group(1).strip())
    return steps

# Build enhanced frame list with substeps
def build_frames(problem, plan):
    frames = []
    state = problem
    grid_size = problem['grid_size']
    frames.append((state, None))
    for action in plan:
        parts = action.split()
        if parts[0] == 'move_character':
            # direct move
            new_state = {
                'snow': state['snow'].copy(),
                'balls': state['balls'].copy(),
                'ball_size': state['ball_size'].copy(),
                'character': parse_loc(parts[2]),
                'grid_size': grid_size
            }
            frames.append((new_state, action))
            state = new_state
        elif parts[0] == 'move_ball':
            b = parts[1]
            to_coord = parse_loc(parts[4])
            # intermediate mid-state
            from_coord = state['balls'][b]
            mid = ((from_coord[0] + to_coord[0]) / 2,
                   (from_coord[1] + to_coord[1]) / 2)
            mid_state = {
                'snow': state['snow'].copy(),
                'balls': state['balls'].copy(),
                'ball_size': state['ball_size'].copy(),
                'character': state['character'],
                'mid_ball': (b, mid),
                'grid_size': grid_size
            }
            frames.append((mid_state, action))
            # final state
            new_state = {
                'snow': state['snow'].copy(),
                'balls': state['balls'].copy(),
                'ball_size': state['ball_size'].copy(),
                'character': state['character'],
                'grid_size': grid_size
            }
            new_state['balls'][b] = to_coord
            # grow if snow
            if new_state['snow'].get(to_coord, False):
                new_state['ball_size'][b] += 1
                new_state['snow'][to_coord] = False
            frames.append((new_state, action))
            state = new_state
    return frames

# Load
problem = parse_problem(PROBLEM_FILE)
plan = parse_plan(PLAN_FILE)
frames = build_frames(problem, plan)

# Visualization setup
grid_size = problem['grid_size']
interval = 500
is_paused = False

fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(bottom=0.2)
pause_ax = plt.axes([0.45, 0.05, 0.1, 0.075])
pause_btn = Button(pause_ax, 'Pause')

step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
desc_text = ax.text(0.5, -0.05, '', transform=ax.transAxes,
                    fontsize=10, ha='center') if SHOW_DESCRIPTION else None

def draw_person(ax, coord, carrying):
    x, y = coord_to_plot(coord, grid_size)
    ax.add_patch(patches.Circle((x,y-0.1),0.1,facecolor='black'))
    ax.add_patch(patches.Rectangle((x-0.05,y-0.1),0.1,0.2,facecolor='black'))
    for dx, dy in [(-0.05,-0.05),(0.05,-0.05)]:
        ax.plot([x+dx,x+2*dx],[y,y+dy],'k-',lw=2)
    for dx, dy in [(-0.05,0.1),(0.05,0.1)]:
        ax.plot([x+dx,x+dx],[y+dy,y+0.2],'k-',lw=2)
    if carrying:
        ax.add_patch(patches.Circle((x+0.15,y-0.1),SMALL_RADIUS,
                                    facecolor='white',edgecolor='navy',lw=2))

def update(i):

    state, action = frames[i]
    ax.clear()
    ax.axis('off')
    # draw grid
    for (r,c), snow in state['snow'].items():
        x,y = coord_to_plot((r,c),grid_size)
        col = 'lightcyan' if snow else 'lightgreen'
        ax.add_patch(patches.Rectangle((x-0.5,y-0.5),1,1,facecolor=col,edgecolor='gray'))
    # draw balls
    cell_balls = {}
    for b, coord in state['balls'].items():
        cell_balls.setdefault(coord,[]).append(b)
    # include mid_ball
    if 'mid_ball' in state:
        b, midc = state['mid_ball']
        cell_balls.setdefault(midc,[]).append(b)
    for coord, bs in cell_balls.items():
        x,y = coord_to_plot(coord,grid_size)
        bs_sorted = sorted(bs, key=lambda b: state['ball_size'][b], reverse=True)
        for idx, b in enumerate(bs_sorted):
            size = state['ball_size'][b]
            radius = [SMALL_RADIUS, MEDIUM_RADIUS, LARGE_RADIUS][size]
            y_off = -0.3 + idx*(radius*2 - 0.05)
            ax.add_patch(patches.Circle((x,y+y_off),radius,facecolor='white',edgecolor='navy',lw=2))
    # carrying true on real frames
    carrying = action and action.startswith('move_ball')
    draw_person(ax, state['character'], carrying)
    # Calcola un indice di step “pulito” (skip frame 0 e mid-frames)
    # Stampiamo solo se c'è un'azione reale e non è mid_ball
    if SHOW_DESCRIPTION and action and 'mid_ball' not in state:
        step_text.set_text(f"Step {update.real_step}/{len(plan)}")
        desc_text.set_text(action)
        print(f"[Step {update.real_step}/{len(plan)}] {action}")
        update.real_step += 1
    else:
        # Per i mid-frame o frame0, mantieni il contatore fisso
        step_text.set_text(f"Step {update.real_step}/{len(plan)}")

    return ax.patches + [step_text] + ([desc_text] if SHOW_DESCRIPTION else [])

update.real_step = 0

ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)

def toggle_pause(event):
    global is_paused
    if is_paused:
        ani.event_source.start(); pause_btn.label.set_text('Pause')
    else:
        ani.event_source.stop(); pause_btn.label.set_text('Play')
    is_paused = not is_paused

pause_btn.on_clicked(toggle_pause)

plt.show()