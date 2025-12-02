"""Pick and Place Execution"""

import time
import numpy as np
from spatialmath import SE3


class PickAndPlaceExecutor:
    """Executes pick-and-place operations using RRMC."""
    
    def __init__(self, robot, env, rrmc_controller, object_manager, 
                 sleep_dt=0.02, update_freq=10):
        """
        Initialize executor.
        
        Args:
            robot: Robot model
            env: RobotEnvironment instance
            rrmc_controller: RRMCController instance
            object_manager: ObjectManager instance
            sleep_dt: Sleep time between updates
            update_freq: Frequency of visualization updates
        """
        self.robot = robot
        self.env = env
        self.rrmc_controller = rrmc_controller
        self.object_manager = object_manager
        self.sleep_dt = sleep_dt
        self.update_freq = update_freq
        
    def execute_single(self, name, trajectory_data):
        """
        Execute pick-and-place for a single object.
        
        Args:
            name: Object name (color)
            trajectory_data: Dictionary with poses and IK solutions
        """
        cube = self.object_manager.cubes[name]
        poses = trajectory_data["poses"]
        q_home = trajectory_data["ik_solutions"]["home"]
        
        try:
            q_current = q_home.copy()
            
            # Phase 1: Home -> pick_above
            print(f"[RRMC] {name}: Moving to pick_above")
            q_current = self._rrmc_move_with_viz(poses["pick_above"], q_current)
            
            # Phase 2: pick_above -> pick (descending)
            print(f"[RRMC] {name}: Descending to pick")
            q_current = self._rrmc_move_with_viz(poses["pick"], q_current)
            
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
            
            # Phase 5: place_above -> place (placing)
            print(f"[RRMC] {name}: Descending to place")
            q_current = self._rrmc_move_with_viz(poses["place"], q_current, 
                                                  cube_attached=True, T_rel=T_rel, cube=cube)
            
            # Set final cube position
            # place_pos = self.object_manager.plate_positions[name]
            # cube.T = SE3(place_pos[0], place_pos[1], self.object_manager.cube_center_z)
            bucket_pos = self.object_manager.buckets_positions[name]
            # cube.T = SE3(bucket_pos[0], bucket_pos[1], bucket_pos[2] - self.object_manager.cube_size/2)
            cube.T = SE3(bucket_pos[0], bucket_pos[1], bucket_pos[2])

            # Phase 6: place -> place_above (retracting)
            print(f"[RRMC] {name}: Retracting gripper")
            q_current = self._rrmc_move_with_viz(poses["place_above"], q_current)
            
            # Phase 7: place_above -> home (returning)
            print(f"[RRMC] {name}: Returning to home")
            q_current = self._rrmc_move_with_viz(poses["home"], q_current)
            
            print(f"[RRMC] {name}: Completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to execute pick-and-place for {name}: {e}")
            
    def _rrmc_move_with_viz(self, target_pose, q_start, cube_attached=False, T_rel=None, cube=None):
        """
        Execute RRMC motion with visualization using natural control loop.
        
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
        
        # Position error
        error = target_pose.t - self.robot.fkine(self.robot.q).t
        
        # Control loop: continue until error is small enough
        while np.linalg.norm(error) >= self.rrmc_controller.position_tol:
            
            # Compute position error to target
            error = target_pose.t - self.robot.fkine(self.robot.q).t
            
            # Compute Jacobian (only position, 3 rows)
            J = self.robot.jacob0(self.robot.q)[0:3, :]
            
            # Damped pseudo-inverse
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + self.rrmc_controller.lambda_damping * np.eye(3))
            
            # Compute joint velocities
            q_dot = self.rrmc_controller.gain * J_damped_pinv @ error
            
            # Apply to robot
            self.robot.qd = q_dot
            
            # Update visualization
            self.env.set_robot_config(self.robot.q)
            if cube_attached and T_rel is not None and cube is not None:
                T_ee = self.robot.fkine(self.robot.q)
                cube.T = T_ee * T_rel
            self.env.step(self.rrmc_controller.dt)
            time.sleep(self.sleep_dt)
                
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
