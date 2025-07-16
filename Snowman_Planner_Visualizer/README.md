# Snowman Planner Visualizer

The **Snowman Planner Visualizer** is a sophisticated tool designed to parse, analyze, and visualize planning solutions for PDDL (Planning Domain Definition Language) problems. It supports both numeric and classic PDDL domains, providing 2D and 3D visualizations of planning outcomes, along with comparative analysis of plan metrics. This project is ideal for researchers, AI planners, and developers working with automated planning systems.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Visualization](#visualization)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features
- **PDDL Parsing and Analysis**: Processes PDDL domain and problem files (both numeric and classic formats) to extract planning solutions.
- **Plan Comparison**: Generates metrics to compare plans (e.g., cost, steps) and outputs results in CSV format.
- **Visualization**:
  - **2D Visualizer**: Displays plan execution in a 2D environment for quick insights.
  - **3D Visualizer**: Provides an interactive 3D rendering of planning outcomes via a web-based interface.
- **Web Interface**: A user-friendly frontend for interacting with visualizations and plan outputs.
- **Extensible Design**: Modular backend and frontend architecture for easy customization and extension.

## Project Structure
```
Snowman_Planner_Visualizer/
├── src/                    # Python source code
│   ├── backend/           # Backend logic for parsing and analysis
│   │   ├── comparasion.py # Plan comparison logic
│   │   ├── visualizer.py  # 2D visualization module
│   │   └── server.py      # 3D visualization server
│   └── frontend/          # Frontend assets and scripts
│       ├── static/        # Static files (CSS, JS, images)
│       │   ├── css/       # Stylesheets
│       │   ├── js/        # JavaScript for interactivity
│       │   └── img/       # Image assets
│       └── templates/     # HTML templates
│           └── index.html # Main web interface
├── pddl/                  # PDDL domain and problem files
│   ├── domains/           # PDDL domain definitions
│   │   ├── domain-numeric.pddl
│   │   └── domain-classic.pddl
│   └── problems/          # PDDL problem instances
│       ├── problem-numeric.pddl
│       └── problem-classic.pddl
├── plans/                 # Output plan files
│   ├── plan_numeric.txt   # Numeric plan output
│   └── plan_classic.txt   # Classic plan output
├── data/                  # Generated data (e.g., CSV, JSON)
│   ├── comparison_metrics.csv # Plan comparison metrics
├── tests/                 # Unit tests
│   ├── test_comparasion.py # Tests for comparison module
│   └── test_visualizer.py  # Tests for visualization module
├── docs/                  # Documentation
│   ├── README.md          # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
└── setup.py               # Project setup script
```

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/Snowman_Planner_Visualizer.git
   cd Snowman_Planner_Visualizer
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Install the Project**:
   ```bash
   python setup.py install
   ```

## Usage
1. **Prepare PDDL Files**:
   - Place your PDDL domain and problem files in `pddl/domains/` and `pddl/problems/`, respectively.
   - Example files (`domain-numeric.pddl`, `problem-numeric.pddl`, `domain-classic.pddl`, `problem-classic.pddl`) are provided.

2. **Run the Backend**:
   - To generate plans and comparison metrics:
     ```bash
     python src/backend/comparasion.py
     ```
   - Output plans are saved in `plans/` (e.g., `plan_numeric.txt`, `plan_classic.txt`).
   - Comparison metrics are saved in `data/comparison_metrics.csv`.

3. **Start the Visualization Server**:
   - For 2D visualization:
     ```bash
     python src/backend/visualizer.py
     ```
   - For 3D visualization (web-based):
     ```bash
     python src/backend/server.py
     ```
   - Access the web interface at `http://localhost:5000` (default port).

## Visualization
- **2D Visualizer**: Run `visualizer.py` to display a 2D representation of the plan execution. Ideal for quick debugging and analysis.
- **3D Visualizer**: Run `server.py` to launch a web server hosting an interactive 3D visualization. Access it via a browser to explore plan steps in a 3D environment.
- **Frontend**: The `index.html` template in `src/frontend/templates/` provides a clean interface for interacting with visualizations and viewing metrics.

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## License
This project is licensed under the MSC AI and CS students of UINCAL License.