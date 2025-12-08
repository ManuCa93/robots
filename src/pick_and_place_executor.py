"""Pick and Place Execution"""

import time
import numpy as np
from spatialmath import SE3


class PickAndPlaceExecutor:
    """Executes pick-and-place operations using RRMC."""
    
    def __init__(self, robot, env, rrmc_controller, object_manager, 
                 sleep_dt=0.005, update_freq=10, gravity_acceleration=0.5):
        """
        Initialize executor.
        
        Args:
            robot: Robot model
            env: RobotEnvironment instance
            rrmc_controller: RRMCController instance
            object_manager: ObjectManager instance
            sleep_dt: Sleep time between updates
            update_freq: Frequency of visualization updates
            gravity_acceleration: Gravity acceleration for cube drop (m/s²)
        """
        self.robot = robot
        self.env = env
        self.rrmc_controller = rrmc_controller
        self.object_manager = object_manager
        self.sleep_dt = sleep_dt
        self.update_freq = update_freq
        self.gravity_acceleration = gravity_acceleration
        
    def execute_single(self, name, trajectory_data):
        """
        Execute pick-and-place for a single object.
        
        Args:
            name: Object name (color)
            trajectory_data: Dictionary with poses and IK solutions
        """
        cube = self.object_manager.cubes[name]
        ik_solutions = trajectory_data["ik_solutions"]
        
        # Generate target poses from IK solutions using fkine
        poses = {key: self.robot.fkine(q) for key, q in ik_solutions.items()}
        
        q_home = ik_solutions["home"]
        
        try:
            q_current = q_home.copy()
            
            # Phase 1: Home -> pick_above (tracking moving cube)
            print(f"[RRMC] {name}: Moving to pick_above (tracking moving target)")
            q_current = self._rrmc_move_with_viz_tracking(poses["pick_above"], q_current, name, phase="pick_above")
            
            # Phase 2: pick_above -> pick (descending while tracking)
            print(f"[RRMC] {name}: Descending to pick (tracking moving target)")
            q_current = self._rrmc_move_with_viz_tracking(poses["pick"], q_current, name, phase="pick")
            
            # Mark cube as picked (stops its circular motion)
            self.object_manager.mark_cube_picked(name)
            
            # Calculate relative transform for attachment
            T_ee_contact = self.robot.fkine(q_current)
            T_cube_contact = cube.T
            T_rel = T_ee_contact.inv() * T_cube_contact

            # Phase 3: pick -> pick_above (lifting with cube)
            print(f"[RRMC] {name}: Lifting with cube")
            q_current = self._rrmc_move_with_viz(poses["pick_above"], q_current, 
                                                  cube_attached=True, T_rel=T_rel, cube=cube)
            
            # Phase 4: pick_above -> place_above (transporting)
            print(f"[RRMC] {name}: Transporting to place_above")
            q_current = self._rrmc_move_with_viz(poses["place_above"], q_current, 
                                                  cube_attached=True, T_rel=T_rel, cube=cube)
            
            # Phase 5: Release cube and simulate gravity drop
            print(f"[RRMC] {name}: Releasing cube with gravity")
            place_pos = self.object_manager.object_place_positions.get(
                name,
                self.object_manager.buckets_positions.get(self.object_manager.get_base_color(name))
            )
            
            # Mark cube as released BEFORE dropping (stops circular motion permanently)
            self.object_manager.mark_cube_released(name)
            
            if place_pos is not None:
                # Update the stored cube position to the place position
                with self.object_manager.motion_lock:
                    self.object_manager.cube_positions[name] = place_pos.copy()
                
                # Get current cube position (at place_above height)
                cube_transform = cube.T
                if hasattr(cube_transform, 't'):
                    current_cube_pos = cube_transform.t
                else:
                    # If it's already an SE3, get translation
                    current_cube_pos = SE3(cube_transform).t
                # Simulate gravity drop to final position
                self._simulate_gravity_drop(cube, current_cube_pos, place_pos)

            # Phase 6: Retracting gripper (no need to move, already at place_above)
            print(f"[RRMC] {name}: Gripper released")
            
            # Phase 7: place_above -> home (returning)
            print(f"[RRMC] {name}: Returning to home")
            q_current = self._rrmc_move_with_viz(poses["home"], q_current)
            
            print(f"[RRMC] {name}: Completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to execute pick-and-place for {name}: {e}")
    
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
            
    def _rrmc_move_with_viz_tracking(self, initial_target_pose, q_start, cube_name, phase="pick_above"):
        """
        Execute RRMC motion while tracking a moving cube.
        
        Args:
            initial_target_pose: Initial target SE3 pose
            q_start: Starting joint configuration
            cube_name: Name of the cube to track
            phase: Phase of motion ("pick_above" or "pick")
            
        Returns:
            Final joint configuration
        """
        # Set initial configuration
        self.robot.q = q_start.copy()
        
        iteration = 0
        max_iterations = 5000
        
        approach_height = 0.12 if phase == "pick_above" else 0.0
        
        # Control loop with dynamic target tracking
        while iteration < max_iterations:
            # Get current cube position (thread-safe)
            current_cube_pos = self.object_manager.get_current_cube_position(cube_name)
            
            if current_cube_pos is not None:
                # Update target pose based on current cube position
                target_pose = SE3(current_cube_pos[0], current_cube_pos[1], 
                                 current_cube_pos[2] + approach_height) * SE3.Rx(np.pi)
            else:
                target_pose = initial_target_pose
            
            # Get current pose
            current_pose = self.robot.fkine(self.robot.q)
            
            # Position error
            pos_error = target_pose.t - current_pose.t
            
            # Orientation error
            R_current = current_pose.R
            R_target = target_pose.R
            R_error = R_target @ R_current.T
            ori_error = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / 2.0
            
            # Check convergence
            pos_error_norm = np.linalg.norm(pos_error)
            ori_error_norm = np.linalg.norm(ori_error)
            
            if pos_error_norm < self.rrmc_controller.position_tol and ori_error_norm < self.rrmc_controller.orientation_tol:
                break
            
            # 6D error
            error_6d = np.concatenate([pos_error, ori_error])
            
            # Full Jacobian
            J = self.robot.jacob0(self.robot.q)
            
            # Damped pseudo-inverse
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + self.rrmc_controller.lambda_damping * np.eye(6))
            
            # Joint velocities with higher gain for tracking (2.5x faster)
            tracking_gain = self.rrmc_controller.gain * 2.5
            q_dot = tracking_gain * J_damped_pinv @ error_6d
            
            # Apply to robot
            self.robot.qd = q_dot
            
            # Update visualization
            self.env.set_robot_config(self.robot.q)
            self.env.step(self.rrmc_controller.dt)
            time.sleep(self.sleep_dt)
            
            iteration += 1
                
        return self.robot.q
    
    def _rrmc_move_with_viz(self, target_pose, q_start, cube_attached=False, T_rel=None, cube=None):
        """
        Execute RRMC motion with visualization using 6-DOF control.
        
        Args:
            target_pose: Target SE3 pose
            q_start: Starting joint configuration
            cube_attached: Whether cube is attached
            T_rel: Relative transform for cube
            cube: Cube object to move
            
        Returns:
            Final joint configuration
        """
        # Set initial configuration
        self.robot.q = q_start.copy()
        
        iteration = 0
        max_iterations = 5000
        
        # Control loop with 6-DOF control
        while iteration < max_iterations:
            # Get current pose
            current_pose = self.robot.fkine(self.robot.q)
            
            # Position error
            pos_error = target_pose.t - current_pose.t
            
            # Orientation error
            R_current = current_pose.R
            R_target = target_pose.R
            R_error = R_target @ R_current.T
            ori_error = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / 2.0
            
            # Check convergence
            pos_error_norm = np.linalg.norm(pos_error)
            ori_error_norm = np.linalg.norm(ori_error)
            
            if pos_error_norm < self.rrmc_controller.position_tol and ori_error_norm < self.rrmc_controller.orientation_tol:
                break
            
            # 6D error
            error_6d = np.concatenate([pos_error, ori_error])
            
            # Full Jacobian
            J = self.robot.jacob0(self.robot.q)
            
            # Damped pseudo-inverse
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + self.rrmc_controller.lambda_damping * np.eye(6))
            
            # Joint velocities
            q_dot = self.rrmc_controller.gain * J_damped_pinv @ error_6d
            
            # Apply to robot
            self.robot.qd = q_dot
            
            # Update visualization
            self.env.set_robot_config(self.robot.q)
            if cube_attached and T_rel is not None and cube is not None:
                T_ee = self.robot.fkine(self.robot.q)
                cube.T = T_ee * T_rel
            self.env.step(self.rrmc_controller.dt)
            time.sleep(self.sleep_dt)
            
            iteration += 1
                
        return self.robot.q
        
    def execute_all(self, trajectories):
        """
        Execute pick-and-place for all objects.
        
        Args:
            trajectories: Dictionary mapping names to trajectory data
        """
        for idx, (name, traj_data) in enumerate(trajectories.items(), 1):
            print(f"\n[INFO] [{idx}/{len(trajectories)}] Executing pick-and-place for {name}")
            self.execute_single(name, traj_data)
