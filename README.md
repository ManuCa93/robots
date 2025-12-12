# Robot Pick-and-Place Simulation

## Overview
This project simulates a robotic pick-and-place operation using two different control methods: **Joint Space Control** and **Resolved Rate Motion Control (RRMC)**. The simulation involves planning and executing trajectories for a robotic arm to pick up objects and place them in designated locations, with dynamic tracking and visualization.

---

## Features
- **Control Methods**:
  - **Joint Space Control**: Smooth trajectory planning in joint space using cubic splines.
  - **RRMC**: Cartesian velocity control with resolved rate motion control.
- **Dynamic Object Tracking**: Real-time adjustments to track moving objects.
- **Trajectory Planning**: Automated planning of pick-and-place trajectories.
- **Visualization**: End-effector paths, joint angles, and workspace overview.
- **Simulation Environment**: Includes object creation, motion, and gravity simulation.

---

## Project Structure
```
robot/
├── main.py                  # Entry point for the simulation
├── README.md                # Project documentation
├── src/                     # Source code directory
│   ├── joint_space_controller.py  # Joint space trajectory control
│   ├── joint_space_executor.py    # Joint space execution logic
│   ├── rrmc_controller.py         # RRMC control logic
│   ├── object_manager.py          # Object creation and management
│   ├── pick_and_place_executor.py # RRMC pick-and-place execution
│   ├── recorder.py                # Data recording for visualization
│   ├── robot_environment.py       # Robot simulation environment
│   ├── trajectory_planner.py      # Trajectory planning logic
│   ├── visualizer.py              # Visualization utilities
│   └── __init__.py                # Package initialization
└── cubes_plot/             # Visualization outputs
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ManuCa93/robots.git
   cd robots
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Run the simulation:
   ```bash
   python main.py
   ```
2. Choose the control method by modifying the `CONTROL_METHOD` variable in `main.py`:
   - `"joint_space"` for Joint Space Control
   - `"rrmc"` for Resolved Rate Motion Control

---

## Key Components
### 1. **Joint Space Control**
- Smooth trajectory planning using cubic splines.
- Direct control of joint angles.
- Suitable for tasks requiring smooth and predictable motion.

### 2. **Resolved Rate Motion Control (RRMC)**
- Cartesian velocity control with damping to handle singularities.
- Real-time adjustments for dynamic environments.
- Suitable for precise Cartesian control.

### 3. **Visualization**
- End-effector paths and joint angles over time.
- Workspace overview showing object positions and trajectories.

---

## Examples
### Joint Space Control
![Joint Space Control Example](cubes_plot/joint_space_example.png)

### RRMC Control
![RRMC Control Example](cubes_plot/rrmc_example.png)

---

## Contributing
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

---

## Acknowledgments
- **Libraries Used**:
  - `numpy`: Numerical computations
  - `scipy`: Interpolation and optimization
  - `spatialmath`: Robotics transformations
  - `roboticstoolbox`: Environment
- **Inspiration**: Robotics control and simulation techniques.