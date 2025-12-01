"""Resolved Rate Motion Control (RRMC) Implementation"""

import numpy as np


class RRMCController:
    """Implements RRMC for Cartesian velocity control with natural motion."""
    
    def __init__(self, dt=0.01, position_tol=0.02, lambda_damping=0.15, gain=1.5):
        """
        Initialize RRMC controller.
        
        Args:
            dt: Time step for integration
            position_tol: Position error tolerance (m)
            lambda_damping: Damping factor for pseudo-inverse
            gain: Proportional gain for velocity control
        """
        self.dt = dt
        self.position_tol = position_tol
        self.lambda_damping = lambda_damping
        self.gain = gain
        
    def move_to_pose(self, robot, target_pose, q_start):
        """
        Move robot to target pose using RRMC with natural convergence.
        Simple control loop that continues until error is small enough.
        
        Args:
            robot: Robot model
            target_pose: SE3 target pose
            q_start: Starting joint configuration
            
        Returns:
            q_final: Final joint configuration
            qd_hist: History of joint velocities
            cond_hist: History of condition numbers
        """
        # Set initial configuration
        robot.q = q_start.copy()
        
        # Position error
        error = target_pose.t - robot.fkine(robot.q).t
        
        # Log variables
        qd_hist = []
        cond_hist = []
        
        # Control loop: continue until error is small enough
        while np.linalg.norm(error) >= self.position_tol:
            
            # Compute position error to target
            error = target_pose.t - robot.fkine(robot.q).t
            
            # Compute Jacobian (only position, 3 rows)
            J = robot.jacob0(robot.q)[0:3, :]
            
            # Damped pseudo-inverse
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + self.lambda_damping * np.eye(3))
            
            # Condition number for monitoring
            cond_number = np.linalg.cond(J_damped_pinv)
            
            # Compute joint velocities
            q_dot = self.gain * J_damped_pinv @ error
            
            # Apply to robot (robot automatically integrates qd)
            robot.qd = q_dot
            
            # Log
            qd_hist.append(robot.qd.copy())
            cond_hist.append(cond_number)
        
        return robot.q, qd_hist, cond_hist
