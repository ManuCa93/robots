"""Joint-Space Pick and Place Execution"""

import time
import numpy as np
from spatialmath import SE3


class JointSpaceExecutor:
    """Executes pick-and-place operations using joint-space trajectories."""
    
    def __init__(self, robot, env, joint_space_controller, object_manager, sleep_dt=0.02, gravity_acceleration=0.5):
        """
        Initialize executor.
        
        Args:
            robot: Robot model
            env: RobotEnvironment instance
            joint_space_controller: JointSpaceController instance
            object_manager: ObjectManager instance
            sleep_dt: Sleep time between trajectory points
            gravity_acceleration: Gravity acceleration for cube drop (m/s²)
        """
        self.robot = robot
        self.env = env
        self.controller = joint_space_controller
        self.object_manager = object_manager
        self.sleep_dt = sleep_dt
        self.gravity_acceleration = gravity_acceleration
        
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
            
            # 5) Release cube and simulate gravity drop
            print(f"[Joint-Space] {name}: Releasing cube with gravity")
            place_pos = self.object_manager.object_place_positions.get(
                name,
                self.object_manager.buckets_positions.get(self.object_manager.get_base_color(name))
            )
            if place_pos is not None:
                # Get current cube position (at place_above height)
                cube_transform = cube.T
                if hasattr(cube_transform, 't'):
                    current_cube_pos = cube_transform.t
                else:
                    current_cube_pos = SE3(cube_transform).t
                # Simulate gravity drop to final position
                self._simulate_gravity_drop(cube, current_cube_pos, place_pos)
            
            # 6) Gripper released (already at place_above, no need to retract)
            print(f"[Joint-Space] {name}: Gripper released")
            
            # 7) place_above -> home (robot returns to home)
            print(f"[Joint-Space] {name}: Returning to home")
            traj_home = trajs["place_abv_to_home"]
            print(f"[DEBUG] Trajectory shape: {traj_home.shape}, points: {len(traj_home)}")
            for idx, q in enumerate(traj_home):
                self.robot.q = q
                self.env.set_robot_config(q)
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
                if idx % 20 == 0:
                    print(f"[DEBUG] Step {idx}/{len(traj_home)}")
            
            print(f"[Joint-Space] {name}: Completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to execute joint-space pick-and-place for {name}: {e}")
    
    def _simulate_gravity_drop(self, cube, start_pos, target_pos):
        """
        Simulate gravity drop of cube from current position to target.
        
        Args:
            cube: Cube object to drop
            start_pos: Starting position (x, y, z)
            target_pos: Target position (x, y, z)
        """
        print(f"[GRAVITY] Start pos: {start_pos}")
        print(f"[GRAVITY] Target pos: {target_pos}")
        
        # Calculate drop distance
        drop_distance = start_pos[2] - target_pos[2]
        print(f"[GRAVITY] Drop distance: {drop_distance:.4f}m")
        
        if drop_distance <= 0:
            # Already at or below target, just set position
            print(f"[GRAVITY] No drop needed (distance <= 0)")
            cube.T = SE3(target_pos[0], target_pos[1], target_pos[2])
            return
        
        # Physics: v² = u² + 2as, where u=0 (initial velocity)
        # Time to fall: t = sqrt(2*h/g)
        fall_time = np.sqrt(2 * drop_distance / self.gravity_acceleration)
        
        # Number of steps for smooth animation
        num_steps = max(int(fall_time / self.sleep_dt), 10)
        
        print(f"[GRAVITY] Fall time: {fall_time:.3f}s, Steps: {num_steps}")
        
        # Simulate falling with increasing velocity
        for step in range(num_steps + 1):
            t = (step / num_steps) * fall_time
            
            # Calculate height using kinematic equation: h = h0 - 0.5*g*t²
            current_z = start_pos[2] - 0.5 * self.gravity_acceleration * t**2
            
            # Ensure we don't go below target
            current_z = max(current_z, target_pos[2])
            
            # Update cube position (x, y stay the same, only z changes)
            cube.T = SE3(target_pos[0], target_pos[1], current_z)
            
            # Update visualization
            self.env.step(self.sleep_dt)
            time.sleep(self.sleep_dt)
            
            # Stop if we've reached the target
            if current_z <= target_pos[2]:
                break
        
        # Ensure final position is exactly at target
        cube.T = SE3(target_pos[0], target_pos[1], target_pos[2])
    
    def execute_all(self, trajectories_dict):
        """
        Execute pick-and-place for all objects.
        
        Args:
            trajectories_dict: Dictionary mapping names to trajectory data
        """
        for idx, (name, traj_data) in enumerate(trajectories_dict.items(), 1):
            print(f"\n[INFO] [{idx}/{len(trajectories_dict)}] Executing pick-and-place for {name}")
            self.execute_single(name, traj_data)
