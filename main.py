"""Main execution script for robot pick-and-place task"""

import numpy as np
import time
from src.robot_environment import RobotEnvironment
from src.object_manager import ObjectManager
from src.rrmc_controller import RRMCController
from src.joint_space_controller import JointSpaceController
from src.trajectory_planner import TrajectoryPlanner
from src.pick_and_place_executor import PickAndPlaceExecutor
from src.joint_space_executor import JointSpaceExecutor
from src.visualizer import Visualizer
from src.recorder import DataRecorder  

def main():
    """Main execution function."""
    
    # Configuration
    CUBE_SIZE = 0.03
    PLATE_SIZE = 0.1
    APPROACH_HEIGHT = 0.12
    cube_height = CUBE_SIZE
    cube_center_z = CUBE_SIZE / 2  
    pick_z = cube_center_z + cube_height / 2  
    
    # Choose control method: "rrmc" or "joint_space"
    CONTROL_METHOD = "joint_space" 
    
    # Generate cubes
    cube_positions_only = {
        'red1':    np.array([0.5,  0.31, pick_z]),
        'red2':    np.array([0.5,  0.19, pick_z]),
        'green1':  np.array([0.5,  0.07, pick_z]),
        'green2':  np.array([0.5, -0.05, pick_z]),
        'yellow1': np.array([0.5, -0.17, pick_z]),
        'blue1':   np.array([0.5, -0.29, pick_z]),
    }
    
    # 1. Initialize environment
    print("[INFO] Initializing robot environment...")
    env = RobotEnvironment()
    env.launch(realtime=True)

    # 1.5 Initialize Data Recorder
    recorder = DataRecorder() # <--- Creazione Recorder

    # 2. Create objects
    print("[INFO] Creating objects...")
    obj_manager = ObjectManager(cube_size=CUBE_SIZE, plate_size=PLATE_SIZE)
    obj_manager.create_cubes(cube_positions_only, env)
    obj_manager.create_buckets(env.terrain_bounds, env, cube_height)
    
    # 2.5 Start circular motion
    print("\n[INFO] Starting circular motion of cubes...")
    obj_manager.start_circular_motion(env)
    time.sleep(1.0) 
    
    # 3. Plan trajectories
    print("\n[INFO] Planning trajectories...")
    
    if CONTROL_METHOD == "joint_space":
        js_controller = JointSpaceController(duration=2.0, n_points=100)
        planner = TrajectoryPlanner(env.panda, env.q_home, approach_height=APPROACH_HEIGHT, 
                                     joint_space_controller=js_controller)
    else:
        planner = TrajectoryPlanner(env.panda, env.q_home, approach_height=APPROACH_HEIGHT)
    
    pick_place_pairs = obj_manager.get_pick_place_pairs()
    trajectories = planner.plan_all_trajectories(pick_place_pairs)
    
    if not trajectories:
        print("[ERROR] No valid trajectories computed. Exiting.")
        return
    
    # 4. Execute pick-and-place
    print(f"\n[INFO] Starting pick-and-place execution using {CONTROL_METHOD.upper()}...")
    
    if CONTROL_METHOD == "joint_space":
        joint_trajs = {name: data.get("joint_trajectories", {}) for name, data in trajectories.items()}
        
        executor = JointSpaceExecutor(env.panda, env, js_controller, obj_manager, 
                                      recorder=recorder, sleep_dt=0.02)
        executor.execute_all(joint_trajs)
    else:
        rrmc = RRMCController(dt=0.01, position_tol=0.005, orientation_tol=0.02, lambda_damping=0.1, gain=8.0)
        
        executor = PickAndPlaceExecutor(env.panda, env, rrmc, obj_manager, 
                                        recorder=recorder, sleep_dt=0.005)
        executor.execute_all(trajectories)
    
    # 5. Stop circular motion
    obj_manager.stop_circular_motion()
    
    # 6. Visualize results
    print("\n[INFO] Generating visualization...")
    
    # Recupera i dati dal recorder
    history_data = recorder.get_data()
    
    # End-Effector Path Plot
    Visualizer.plot_end_effector_paths(history_data, title=f"Real End-Effector Trajectories ({CONTROL_METHOD})")
    
    # Overview Final
    Visualizer.plot_workspace_overview(obj_manager)
    
    print("\n[SUCCESS] All operations completed successfully!")

if __name__ == "__main__":
    main()