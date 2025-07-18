import re

def parse_loc(loc):
    parts = loc.split('_')
    return int(parts[1])-1, int(parts[2])-1

def parse_problem_basic(content):
    balls = {}
    for match in re.finditer(r"\(ball_at (\S+) (\S+)\)", content):
        ball, loc = match.groups()
        balls[ball] = parse_loc(loc)
    return balls

def parse_plan_basic(content):
    steps = []
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith(';'):
            steps.append(line)
    return steps