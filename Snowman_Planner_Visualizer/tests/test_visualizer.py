import pytest
from src.backend.visualizer import SnowmanVisualizer

def test_parse_loc():
    viz = SnowmanVisualizer('pddl/problems/problem-numeric.pddl', 'plans/plan_numeric.txt', 'data/plan_data.json')
    assert viz.parse_loc('loc_1_2') == (0, 1)

def test_parse_problem():
    viz = SnowmanVisualizer('pddl/problems/problem-numeric.pddl', 'plans/plan_numeric.txt', 'data/plan_data.json')
    prob = viz.parse_problem()
    assert prob['grid_size'] == 5
    assert prob['snow'][(0, 0)] == True
    assert prob['balls']['ball_0'] == (1, 2)

def test_parse_plan():
    viz = SnowmanVisualizer('pddl/problems/problem-numeric.pddl', 'plans/plan_numeric.txt', 'data/plan_data.json')
    plan = viz.parse_plan()
    assert len(plan) == 92
    assert plan[0].startswith('move_character')
