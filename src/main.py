"""Main execution script for robot pick-and-place task"""

import numpy as np
from robot_environment import RobotEnvironment
from object_manager import ObjectManager
from rrmc_controller import RRMCController
from joint_space_controller import JointSpaceController
from trajectory_planner import TrajectoryPlanner
from pick_and_place_executor import PickAndPlaceExecutor
from joint_space_executor import JointSpaceExecutor
from visualizer import Visualizer


def main():
    """Main execution function."""
    
    # Configuration
    CUBE_SIZE = 0.06
    PLATE_SIZE = 0.1
    APPROACH_HEIGHT = 0.12
    
    # Choose control method: "rrmc" or "joint_space"
    CONTROL_METHOD = "rrmc"  # Change to "joint_space" to use joint-space control
    
    # Define cube pick positions
    cube_pick_positions = {
        'red': np.array([0.5, 0.25, CUBE_SIZE]),
        'blue': np.array([0.5, 0.13, CUBE_SIZE]),
        'green': np.array([0.5, 0.01, CUBE_SIZE]),
        'yellow': np.array([0.5, -0.11, CUBE_SIZE]),
    }
    
    # 1. Initialize environment
    print("[INFO] Initializing robot environment...")
    env = RobotEnvironment()
    env.launch(realtime=True)
    
    # 2. Create objects
    print("[INFO] Creating objects...")
    obj_manager = ObjectManager(cube_size=CUBE_SIZE, plate_size=PLATE_SIZE)
    obj_manager.create_cubes(cube_pick_positions, env)
    obj_manager.create_plates(env.terrain_bounds, env)
    
    # 3. Plan trajectories
    print("\n[INFO] Planning trajectories...")
    
    if CONTROL_METHOD == "joint_space":
        # Use joint-space controller
        js_controller = JointSpaceController(duration=2.0, n_points=100)
        planner = TrajectoryPlanner(env.panda, env.q_home, approach_height=APPROACH_HEIGHT, 
                                     joint_space_controller=js_controller)
    else:
        # Use RRMC (no joint-space controller needed)
        planner = TrajectoryPlanner(env.panda, env.q_home, approach_height=APPROACH_HEIGHT)
    
    pick_place_pairs = obj_manager.get_pick_place_pairs()
    trajectories = planner.plan_all_trajectories(pick_place_pairs)
    
    if not trajectories:
        print("[ERROR] No valid trajectories computed. Exiting.")
        return
    
    # 4. Execute pick-and-place operations based on control method
    print(f"\n[INFO] Starting pick-and-place execution using {CONTROL_METHOD.upper()}...")
    
    if CONTROL_METHOD == "joint_space":
        # Extract joint-space trajectories
        joint_trajs = {name: data["joint_trajectories"] for name, data in trajectories.items()}
        executor = JointSpaceExecutor(env.panda, env, js_controller, obj_manager, sleep_dt=0.02)
        executor.execute_all(joint_trajs)
    else:
        # Use RRMC
        rrmc = RRMCController(dt=0.05, K=0.5)
        executor = PickAndPlaceExecutor(env.panda, env, rrmc, obj_manager, sleep_dt=0.02, update_freq=10)
        executor.execute_all(trajectories)
    
    # 5. Visualize results
    print("\n[INFO] Generating visualization...")
    if CONTROL_METHOD == "rrmc":
        Visualizer.plot_rrmc_waypoints(trajectories)
    Visualizer.plot_workspace_overview(obj_manager)
    
    print("\n[SUCCESS] All operations completed successfully!")


if __name__ == "__main__":
    main()
