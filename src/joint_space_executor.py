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
    
    def _execute_tracking_motion(self, name, trajs, traj_key, phase):
        """
        Execute motion with continuous tracking of moving cube.
        
        Args:
            name: Cube name
            trajs: Trajectory dictionary
            traj_key: Key for trajectory in trajs dict
            phase: \"pick_above\" or \"pick\"
            
        Returns:
            Last joint configuration reached
        """
        max_tracking_time = 10.0  # Maximum time to spend tracking (seconds)
        start_time = time.time()
        convergence_threshold = 0.03  # 3cm convergence threshold
        iteration = 0
        min_iterations = 3  # Always do at least a few iterations
        
        while time.time() - start_time < max_tracking_time:
            iteration += 1
            
            # Get current cube position
            current_cube_pos = self.object_manager.get_current_cube_position(name)
            if current_cube_pos is None:
                print(f"[Joint-Space] No cube position available")
                break
            
            # Get current robot position
            q_current = self.robot.q.copy()
            
            # Compute target based on phase
            if phase == "pick_above":
                z_offset = 0.12
            else:  # pick
                z_offset = 0.01  # 1cm above cube top
            
            target_pos = current_cube_pos + np.array([0, 0, z_offset])
            
            # Check if we're close enough (but always do minimum iterations)
            current_ee_pos = self.robot.fkine(q_current).t
            distance_to_target = np.linalg.norm(current_ee_pos[:2] - target_pos[:2])  # XY distance only
            
            if iteration == 1 or iteration % 5 == 0:
                print(f"[Joint-Space] Iteration {iteration}, distance: {distance_to_target*1000:.1f}mm")
            
            if iteration > min_iterations and distance_to_target < convergence_threshold:
                # Close enough, stop tracking
                print(f"[Joint-Space] Reached target after {iteration} iterations (distance: {distance_to_target*1000:.1f}mm)")
                break
            
            # Compute IK for current target
            target_pose = SE3(target_pos[0], target_pos[1], target_pos[2]) * SE3.Rx(np.pi)
            ik_result = self.robot.ikine_LM(target_pose, q0=q_current)
            
            if not ik_result.success:
                print(f"[Joint-Space] IK failed for tracking, using previous trajectory")
                break
            
            q_target = ik_result.q
            
            # Generate short trajectory segment (0.5 seconds worth)
            segment_duration = 0.5
            segment_points = int(segment_duration / self.sleep_dt)
            trajectory_segment = self.controller.interpolate(q_current, q_target, duration=segment_duration, n_points=segment_points)
            
            # Execute trajectory segment
            for q in trajectory_segment:
                self.robot.q = q
                self.env.set_robot_config(q)
                self.env.step(0.01)
                time.sleep(self.sleep_dt)
        
        return self.robot.q.copy()
    
    def execute_all(self, trajectories_dict):
        """
        Execute pick-and-place for all objects.
        
        Args:
            trajectories_dict: Dictionary mapping names to trajectory data
        """
        for idx, (name, traj_data) in enumerate(trajectories_dict.items(), 1):
            print(f"\n[INFO] [{idx}/{len(trajectories_dict)}] Executing pick-and-place for {name}")
            self.execute_single(name, traj_data)
