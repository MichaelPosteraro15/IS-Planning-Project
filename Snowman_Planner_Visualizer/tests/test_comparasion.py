import pytest
import os
from src.backend.comparasion import parse_problem_for_snow, simulate_plan

def test_parse_problem_for_snow():
    snow = parse_problem_for_snow('pddl/problems/problem-numeric.pddl')
    assert 'loc_1_1' in snow
    assert 'loc_2_3' not in snow

def test_simulate_plan():
    action_counts, ball_growth_count, final_loc, final_sizes, computed_cost = simulate_plan(
        'plans/plan_numeric.txt', 'pddl/problems/problem-numeric.pddl'
    )
    assert action_counts['move_character'] > 0
    assert action_counts['move_ball'] > 0
    assert final_loc == 'loc_1_3'
    assert computed_cost == action_counts['move_character'] + action_counts['move_ball']
