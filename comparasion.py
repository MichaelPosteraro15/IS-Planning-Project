import re
import csv
import glob
import os
import getpass

# Regex for metrics in plan files
METRIC_REGEX = {
    'ground_time_ms': re.compile(r'grounding time:\s*(\d+)'),
    'f_count': re.compile(r'\|f\|:(\d+)'),
    'x_count': re.compile(r'\|x\|:(\d+)'),
    'a_count': re.compile(r'\|a\|:(\d+)'),
    'p_count': re.compile(r'\|p\|:(\d+)'),
    'e_count': re.compile(r'\|e\|:(\d+)'),
    'preproc_ms': re.compile(r'h1 setup time \(msec\):\s*(\d+)'),
    'search_time_ms': re.compile(r'search time \(msec\):\s*(\d+)'),
    'plan_length': re.compile(r'plan-length:(\d+)'),
    'plan_cost': re.compile(r'metric \(search\):([\d\.]+)'),
    'expanded_nodes': re.compile(r'expanded nodes:(\d+)'),
    'evaluated_states': re.compile(r'states evaluated:(\d+)'),
    'dead_ends': re.compile(r'number of dead-ends detected:(\d+)'),
    'duplicates': re.compile(r'number of duplicates detected:(\d+)'),
}

def parse_problem_for_snow(problem_file):
    """Parse problem file to get snowy locations."""
    snow_locations = set()
    try:
        with open(problem_file, 'r') as f:
            for line in f:
                line = line.strip()
                if 'location_type' in line:
                    m = re.match(r'\(= \(location_type (\S+)\) (\d+)\)', line)
                    if m and m.group(2) == '1':
                        snow_locations.add(m.group(1))
                elif 'snow' in line:
                    m = re.match(r'\(snow (\S+)\)', line)
                    if m:
                        snow_locations.add(m.group(1))
    except FileNotFoundError:
        print(f"Warning: Problem file {problem_file} not found.")
        return snow_locations
    return snow_locations

def simulate_plan(log_file, problem_file):
    """Simulate plan to count actions, ball growth, and final state."""
    action_counts = {'move_character': 0, 'move_ball': 0, 'goal': 0}
    ball_growth_count = 0
    ball_locations = {}
    ball_sizes = {}
    snow_locations = parse_problem_for_snow(problem_file)
    final_loc = 'unknown'

    # Parse initial ball positions and sizes
    try:
        with open(problem_file, 'r') as f:
            for line in f:
                line = line.strip()
                if 'ball_at' in line:
                    m = re.match(r'\(ball_at (\S+) (\S+)\)', line)
                    if m:
                        ball_locations[m.group(1)] = m.group(2)
                if 'ball_size' in line:
                    m = re.match(r'\(= \(ball_size (\S+)\) (\d+)\)', line)
                    if m:
                        ball_sizes[m.group(1)] = int(m.group(2))
                    elif 'ball_size_small' in line:
                        m = re.match(r'\(ball_size_small (\S+)\)', line)
                        if m:
                            ball_sizes[m.group(1)] = 0
    except FileNotFoundError:
        print(f"Error: Problem file {problem_file} not found.")
        return action_counts, ball_growth_count, final_loc, '', 0

    # Process plan
    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                for action in action_counts:
                    if re.match(r'^\s*\d+\.\d+:\s*\(' + action + r'\b', line):
                        action_counts[action] += 1
                        if action == 'move_ball':
                            m = re.match(r'^\s*\d+\.\d+:\s*\(move_ball (\S+) \S+ \S+ (\S+) .+\)', line)
                            if m:
                                ball = m.group(1)
                                to_loc = m.group(2)
                                if to_loc in snow_locations and ball_sizes.get(ball, 0) < 2:
                                    ball_growth_count += 1
                                    ball_sizes[ball] = ball_sizes.get(ball, 0) + 1
                                    snow_locations.discard(to_loc)
                                ball_locations[ball] = to_loc
                        elif action == 'goal':
                            m = re.match(r'^\s*\d+\.\d+:\s*\(goal (\S+) (\S+) (\S+) (\S+)\)', line)
                            if m:
                                final_loc = m.group(4)
    except FileNotFoundError:
        print(f"Error: Plan file {log_file} not found.")
        return action_counts, ball_growth_count, final_loc, '', 0

    final_sizes = ','.join(f'{b}:{ball_sizes.get(b, 0)}' for b in sorted(ball_locations.keys()))
    computed_cost = action_counts['move_character'] + action_counts['move_ball']
    return action_counts, ball_growth_count, final_loc, final_sizes, computed_cost

def main():
    out_csv = 'comparison_metrics.csv'
    # Fallback to user home directory if permission denied
    fallback_out_csv = os.path.join(os.path.expanduser('~'), 'out_snowman.csv')
    problem_file_map = {
        'plan_numeric': 'problem-numeric.pddl',
        'plan_classic': 'problem-classic.pddl'
    }

    # Define header outside try block
    header = ['run_name'] + list(METRIC_REGEX.keys()) + [
        'move_character_count', 'move_ball_count', 'goal_count',
        'ball_growth_count', 'final_ball_location', 'final_ball_sizes', 'computed_cost'
    ]

    try:
        with open(out_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for filepath in glob.glob('*.txt'):
                run_name = os.path.splitext(os.path.basename(filepath))[0]
                if run_name not in problem_file_map:
                    print(f"Skipping {filepath}: No matching problem file.")
                    continue
                text = open(filepath, 'r').read()
                row = [run_name]

                for key, regex in METRIC_REGEX.items():
                    m = regex.search(text)
                    row.append(m.group(1) if m else '')

                action_counts, ball_growth_count, final_location, final_sizes, computed_cost = simulate_plan(
                    filepath, problem_file_map[run_name]
                )
                row.extend([
                    action_counts['move_character'],
                    action_counts['move_ball'],
                    action_counts['goal'],
                    ball_growth_count,
                    final_location,
                    final_sizes,
                    computed_cost
                ])

                writer.writerow(row)
        print(f" Summary written to {out_csv}")

    except PermissionError as e:
        print(f"Error: Permission denied when writing to {out_csv}: {e}")
        print(f"Attempting to write to {fallback_out_csv} instead...")
        try:
            with open(fallback_out_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for filepath in glob.glob('*.txt'):
                    run_name = os.path.splitext(os.path.basename(filepath))[0]
                    if run_name not in problem_file_map:
                        print(f"Skipping {filepath}: No matching problem file.")
                        continue
                    text = open(filepath, 'r').read()
                    row = [run_name]
                    for key, regex in METRIC_REGEX.items():
                        m = regex.search(text)
                        row.append(m.group(1) if m else '')
                    action_counts, ball_growth_count, final_location, final_sizes, computed_cost = simulate_plan(
                        filepath, problem_file_map[run_name]
                    )
                    row.extend([
                        action_counts['move_character'],
                        action_counts['move_ball'],
                        action_counts['goal'],
                        ball_growth_count,
                        final_location,
                        final_sizes,
                        computed_cost
                    ])
                    writer.writerow(row)
            print(f" Summary written to {fallback_out_csv}")
        except PermissionError as e:
            print(f"Error: Permission denied when writing to {fallback_out_csv}: {e}")
            print("Troubleshooting steps:")
            print("1. Ensure 'comparison_metrics.csv' is not open in another program (e.g., Excel).")
            print("2. Run PowerShell as Administrator: Right-click PowerShell, select 'Run as administrator'.")
            print("3. Check directory permissions: Right-click the project folder, Properties -> Security, ensure your user has 'Write' access.")
            print(f"4. Manually delete 'comparison_metrics.csv' if it exists: del {out_csv}")
            print(f"5. Try saving to a different location by modifying 'out_csv' in the script (e.g., 'C:\\Users\\{getpass.getuser()}\\Desktop\\comparison_metrics.csv').")

if __name__ == "__main__":
    main()