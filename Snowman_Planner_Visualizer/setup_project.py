import os
import shutil
from uuid import uuid4

# Define the project structure
structure = {
    'Snowman_Planner_Visualizer': [
        'src/backend',
        'src/frontend/static/css',
        'src/frontend/static/js',
        'src/frontend/templates',
        'pddl/domains',
        'pddl/problems',
        'plans',
        'data',
        'tests',
        'docs'
    ]
}

# Files to move (based on provided files)
file_mapping = {
    'comparasion.py': 'src/backend/comparasion.py',
    'server.py': 'src/backend/server.py',
    'visualizer.py': 'src/backend/visualizer.py',
    'index.html': 'src/frontend/templates/index.html',
    'domain-numeric.pddl': 'pddl/domains/domain-numeric.pddl',
    'domain-classic.pddl': 'pddl/domains/domain-classic.pddl',
    'problem-numeric.pddl': 'pddl/problems/problem-numeric.pddl',
    'problem-classic.pddl': 'pddl/problems/problem-classic.pddl',
    'plan_numeric.txt': 'plans/plan_numeric.txt',
    'plan_classic.txt': 'plans/plan_classic.txt'
}

# Create directories
def create_structure():
    base_dir = 'snowman-planner'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for dir_path in structure['snowman-planner']:
        os.makedirs(os.path.join(base_dir, dir_path), exist_ok=True)

# Move existing files
def move_files():
    for src, dest in file_mapping.items():
        if os.path.exists(src):
            dest_path = os.path.join('snowman-planner', dest)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(src, dest_path)
            print(f"Moved {src} to {dest_path}")
        else:
            print(f"Warning: {src} not found")

# Create requirements.txt
def create_requirements():
    requirements = """flask==2.3.3
matplotlib==3.9.2
pytest==8.3.3
"""
    with open('snowman-planner/requirements.txt', 'w') as f:
        f.write(requirements)
    print("Created requirements.txt")

# Create .gitignore
def create_gitignore():
    gitignore = """*.pyc
__pycache__/
*.csv
*.json
data/
plans/*.txt
"""
    with open('snowman-planner/.gitignore', 'w') as f:
        f.write(gitignore)
    print("Created .gitignore")

# Create README.md
def create_readme():
    readme = """# Snowman Planner Visualizer

A project to visualize PDDL-based planning for a snowman-building domain.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask server:
   ```bash
   python src/backend/server.py
   ```

3. Open `http://localhost:8000` in a browser to view the visualizer.

## Directory Structure

- `src/backend/`: Python scripts for parsing and visualization
- `src/frontend/`: HTML, CSS, JS for web interface
- `pddl/`: PDDL domain and problem files
- `plans/`: Plan output files
- `data/`: Generated CSV and JSON files
- `tests/`: Unit tests
- `docs/`: Documentation

## Usage

- Select a plan file from the dropdown in the web interface.
- Use Play, Next, Prev, and Reset buttons to control the animation.
- Adjust speed using the slider.

© Unical 2025/2026 MSc AI and CS students
"""
    with open('snowman-planner/docs/README.md', 'w') as f:
        f.write(readme)
    print("Created README.md")

def main():
    create_structure()
    move_files()
    create_requirements()
    create_gitignore()
    create_readme()

if __name__ == '__main__':
    main()
