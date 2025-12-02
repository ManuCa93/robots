"""Resolved Rate Motion Control (RRMC) Implementation"""

import numpy as np


class RRMCController:
    """Implements RRMC for Cartesian velocity control with natural motion."""
    
    def __init__(self, dt=0.01, position_tol=0.002, orientation_tol=0.01, lambda_damping=0.15, gain=1.5):
        """
        Initialize RRMC controller.
        
        Args:
            dt: Time step for integration
            position_tol: Position error tolerance (m)
            orientation_tol: Orientation error tolerance (rad)
            lambda_damping: Damping factor for pseudo-inverse
            gain: Proportional gain for velocity control
        """
        self.dt = dt
        self.position_tol = position_tol
        self.orientation_tol = orientation_tol
        self.lambda_damping = lambda_damping
        self.gain = gain
        
    def move_to_pose(self, robot, target_pose, q_start, max_iterations=5000):
        """
        Move robot to target pose using RRMC with 6-DOF control (position + orientation).
        
        Args:
            robot: Robot model
            target_pose: SE3 target pose
            q_start: Starting joint configuration
            max_iterations: Maximum iterations
            
        Returns:
            q_final: Final joint configuration
            qd_hist: History of joint velocities
            cond_hist: History of condition numbers
        """
        # Set initial configuration
        robot.q = q_start.copy()
        
        # Log variables
        qd_hist = []
        cond_hist = []
        
        iteration = 0
        
        # Control loop: continue until both position and orientation errors are small
        while iteration < max_iterations:
            # Get current end-effector pose
            current_pose = robot.fkine(robot.q)
            
            # Position error (3D vector)
            pos_error = target_pose.t - current_pose.t
            
            # Orientation error using angle-axis from rotation matrix
            R_current = current_pose.R
            R_target = target_pose.R
            R_error = R_target @ R_current.T
            
            # Convert rotation matrix to angle-axis (simple skew-symmetric extraction)
            # For small angles: omega ≈ [R23-R32, R31-R13, R12-R21] / 2
            ori_error = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / 2.0
            
            # Check convergence
            pos_error_norm = np.linalg.norm(pos_error)
            ori_error_norm = np.linalg.norm(ori_error)
            
            if pos_error_norm < self.position_tol and ori_error_norm < self.orientation_tol:
                break
            
            # 6D error vector [linear_velocity; angular_velocity]
            error_6d = np.concatenate([pos_error, ori_error])
            
            # Full 6D Jacobian
            J = robot.jacob0(robot.q)
            
            # Damped pseudo-inverse
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + self.lambda_damping * np.eye(6))
            
            # Condition number for monitoring
            cond_number = np.linalg.cond(J)
            
            # Compute joint velocities
            q_dot = self.gain * J_damped_pinv @ error_6d
            
            # Apply to robot
            robot.qd = q_dot
            
            # Log
            qd_hist.append(robot.qd.copy())
            cond_hist.append(cond_number)
            
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"[WARNING] RRMC max iterations. Errors: pos={pos_error_norm:.4f}m, ori={ori_error_norm:.4f}rad")
        
        return robot.q, qd_hist, cond_hist
