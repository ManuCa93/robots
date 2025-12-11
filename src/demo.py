"""Demo script showing both RRMC and Joint-Space control methods"""

import numpy as np
from robot_environment import RobotEnvironment
from object_manager import ObjectManager
from rrmc_controller import RRMCController
from joint_space_controller import JointSpaceController
from trajectory_planner import TrajectoryPlanner
from pick_and_place_executor import PickAndPlaceExecutor
from joint_space_executor import JointSpaceExecutor


# Configuration
CUBE_SIZE = 0.06
PLATE_SIZE = 0.1
APPROACH_HEIGHT = 0.12

# Define cube pick positions
cube_pick_positions = {
    'red': np.array([0.5, 0.25, CUBE_SIZE]),
    'blue': np.array([0.5, 0.13, CUBE_SIZE]),
}

# Initialize environment
print("="*60)
print("DEMO: RRMC vs Joint-Space Control")
print("="*60)

env = RobotEnvironment()
env.launch(realtime=True)

# Create objects
obj_manager = ObjectManager(cube_size=CUBE_SIZE, plate_size=PLATE_SIZE)
obj_manager.create_cubes(cube_pick_positions, env)
obj_manager.create_plates(env.terrain_bounds, env)

pick_place_pairs = obj_manager.get_pick_place_pairs()

# ============================================================
# DEMO 1: RRMC Control
# ============================================================
print("\n" + "="*60)
print("DEMO 1: Using RRMC (Resolved Rate Motion Control)")
print("="*60)

planner_rrmc = TrajectoryPlanner(env.panda, env.q_home, approach_height=APPROACH_HEIGHT)
trajectories_rrmc = planner_rrmc.plan_all_trajectories(pick_place_pairs)

rrmc = RRMCController(dt=0.01, position_tol=0.02, lambda_damping=0.15, gain=1.5)
executor_rrmc = PickAndPlaceExecutor(env.panda, env, rrmc, obj_manager)

print("\n[RRMC] Executing pick-and-place for 'red' cube...")
executor_rrmc.execute_single('red', trajectories_rrmc['red'])

# ============================================================
# DEMO 2: Joint-Space Control
# ============================================================
print("\n" + "="*60)
print("DEMO 2: Using Joint-Space Trajectory Control")
print("="*60)

js_controller = JointSpaceController(duration=2.0, n_points=100)
planner_js = TrajectoryPlanner(env.panda, env.q_home, approach_height=APPROACH_HEIGHT,
                                joint_space_controller=js_controller)
trajectories_js = planner_js.plan_all_trajectories(pick_place_pairs)

executor_js = JointSpaceExecutor(env.panda, env, js_controller, obj_manager)

print("\n[Joint-Space] Executing pick-and-place for 'blue' cube...")
executor_js.execute_single('blue', trajectories_js['blue']['joint_trajectories'])

print("\n" + "="*60)
print("DEMO COMPLETE!")
print("="*60)
print("\nComparison:")
print("- RRMC: Cartesian-space control, smooth operational space motion")
print("- Joint-Space: Pre-planned joint trajectories, faster execution")
