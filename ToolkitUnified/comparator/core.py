import re

def parse_all_metrics(content):
    metrics = {}
    patterns = {
        'plan_length': r'plan-length:(\d+)',
        'planning_time': r'planning time \(msec\): (\d+)',
        'search_time': r'search time \(msec\): (\d+)',
        'heuristic_time': r'heuristic time \(msec\): (\d+)',
        'grounding_time': r'grounding time: (\d+)',
        'expanded_nodes': r'expanded nodes:(\d+)',
        'states_evaluated': r'states evaluated:(\d+)',
        'dead_ends': r'number of dead-ends detected:(\d+)',
        'duplicates': r'number of duplicates detected:(\d+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, content, re.IGNORECASE)
        metrics[key] = int(m.group(1)) if m else 0
    return metrics

def parse_plan_actions(content):
    steps = []
    block = re.search(r'found plan:(.*?)(?:plan-length|metric|planning time)', content, re.DOTALL|re.IGNORECASE)
    if block:
        for ln in block.group(1).splitlines():
            ln = ln.strip()
            if not ln or ln.startswith(';'): 
                continue
            ln = re.sub(r'^\d+\.\d+:\s*', '', ln)
            if ln.startswith('(') and ln.endswith(')'):
                steps.append(ln[1:-1].strip())
    
    if not steps:
        raise ValueError("No valid actions found.")
    
    metrics = parse_all_metrics(content)
    if metrics['plan_length'] == 0:
        metrics['plan_length'] = len(steps)
    
    return steps, metrics