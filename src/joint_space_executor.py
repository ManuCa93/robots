"""Joint-Space Pick and Place Execution"""

import time
from spatialmath import SE3


class JointSpaceExecutor:
    """Executes pick-and-place operations using joint-space trajectories."""
    
    def __init__(self, robot, env, joint_space_controller, object_manager, sleep_dt=0.02):
        """
        Initialize executor.
        
        Args:
            robot: Robot model
            env: RobotEnvironment instance
            joint_space_controller: JointSpaceController instance
            object_manager: ObjectManager instance
            sleep_dt: Sleep time between trajectory points
        """
        self.robot = robot
        self.env = env
        self.controller = joint_space_controller
        self.object_manager = object_manager
        self.sleep_dt = sleep_dt
        
    def execute_single(self, name, trajectory_data):
        """
        Execute joint-space pick-and-place for a single object.
        
        Args:
            name: Object name (color)
            trajectory_data: Dictionary with joint-space trajectories
        """
        cube = self.object_manager.cubes[name]
        trajs = trajectory_data
        
        try:
            # 1) Home -> pick_above (robot approaches, cube stationary)
            print(f"[Joint-Space] {name}: Moving to pick_above")
            for q in trajs["home_to_pick_above"]:
                self.env.set_robot_config(q)
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
            
            # 2) pick_above -> pick (descending, cube still stationary)
            print(f"[Joint-Space] {name}: Descending to pick")
            last_q_contact = trajs["pick_above_to_pick"][-1]
            for q in trajs["pick_above_to_pick"]:
                self.env.set_robot_config(q)
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
            
            # Calculate relative transform for attachment
            T_ee_contact = self.robot.fkine(last_q_contact)
            T_cube_contact = cube.T
            T_rel = T_ee_contact.inv() * T_cube_contact
            
            # 3) pick -> pick_above (cube ATTACHED to the gripper)
            print(f"[Joint-Space] {name}: Lifting with cube")
            for q in trajs["pick_to_pick_above"]:
                self.env.set_robot_config(q)
                T_ee = self.robot.fkine(q)
                cube.T = T_ee * T_rel  # Cube follows the hand rigidly
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
            
            # 4) pick_above -> place_above (transport in air with cube attached)
            print(f"[Joint-Space] {name}: Transporting to place_above")
            for q in trajs["pick_above_to_place_abv"]:
                self.env.set_robot_config(q)
                T_ee = self.robot.fkine(q)
                cube.T = T_ee * T_rel
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
            
            # 5) place_above -> place (descending to place the cube)
            print(f"[Joint-Space] {name}: Descending to place")
            for q in trajs["place_abv_to_place"]:
                self.env.set_robot_config(q)
                T_ee = self.robot.fkine(q)
                cube.T = T_ee * T_rel
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
            
            # Force the cube exactly into the place position
            place_pos = self.object_manager.plate_positions[name]
            cube.T = SE3(place_pos[0], place_pos[1], self.object_manager.cube_center_z)
            
            # 6) place -> place_above (gripper rises, cube STAYS on the table)
            print(f"[Joint-Space] {name}: Retracting gripper")
            for q in trajs["place_to_place_abv"]:
                self.env.set_robot_config(q)
                # Do not update cube.T: the cube stays where we left it
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
            
            # 7) place_above -> home (robot returns to home)
            print(f"[Joint-Space] {name}: Returning to home")
            for q in trajs["place_abv_to_home"]:
                self.env.set_robot_config(q)
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
            
            print(f"[Joint-Space] {name}: Completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to execute joint-space pick-and-place for {name}: {e}")
    
    def execute_all(self, trajectories_dict):
        """
        Execute pick-and-place for all objects.
        
        Args:
            trajectories_dict: Dictionary mapping names to trajectory data
        """
        for idx, (name, traj_data) in enumerate(trajectories_dict.items(), 1):
            print(f"\n[INFO] [{idx}/{len(trajectories_dict)}] Executing pick-and-place for {name}")
            self.execute_single(name, traj_data)
