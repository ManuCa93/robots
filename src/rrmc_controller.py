"""Resolved Rate Motion Control (RRMC) Implementation"""

import numpy as np


class RRMCController:
    """Implements RRMC for Cartesian velocity control."""
    
    def __init__(self, dt=0.05, position_tol=0.001, orientation_tol=0.01, 
                 max_iterations=2000, lambda_damping=0.01, K=0.5):
        """
        Initialize RRMC controller.
        
        Args:
            dt: Time step for integration
            position_tol: Position error tolerance (m)
            orientation_tol: Orientation error tolerance (rad)
            max_iterations: Maximum number of iterations
            lambda_damping: Damping factor for pseudo-inverse
            K: Proportional gain
        """
        self.dt = dt
        self.position_tol = position_tol
        self.orientation_tol = orientation_tol
        self.max_iterations = max_iterations
        self.lambda_damping = lambda_damping
        self.K = K
        
    def move_to_pose(self, robot, target_pose, q_start):
        """
        Move robot to target pose using RRMC.
        
        Args:
            robot: Robot model
            target_pose: SE3 target pose
            q_start: Starting joint configuration
            
        Returns:
            q_final: Final joint configuration
            success: Boolean indicating convergence
        """
        q = q_start.copy()
        
        for i in range(self.max_iterations):
            # Current end-effector pose
            T_current = robot.fkine(q)
            
            # Pose error
            T_error = target_pose * T_current.inv()
            position_error = T_error.t
            
            # Orientation error (axis-angle)
            R_error = T_error.R
            angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
            
            if angle < 1e-6:
                orientation_error = np.zeros(3)
            else:
                axis = np.array([
                    R_error[2, 1] - R_error[1, 2],
                    R_error[0, 2] - R_error[2, 0],
                    R_error[1, 0] - R_error[0, 1]
                ]) / (2 * np.sin(angle))
                orientation_error = angle * axis
            
            # Check convergence
            if (np.linalg.norm(position_error) < self.position_tol and 
                np.linalg.norm(orientation_error) < self.orientation_tol):
                return q, True
            
            # Spatial error
            spatial_error = np.concatenate([position_error, orientation_error])
            
            # Jacobian and damped pseudo-inverse
            J = robot.jacob0(q)
            J_damped = J.T @ np.linalg.inv(J @ J.T + self.lambda_damping**2 * np.eye(6))
            
            # Desired Cartesian velocity
            v_desired = self.K * spatial_error
            
            # Joint velocities
            q_dot = J_damped @ v_desired
            
            # Integrate
            q = q + q_dot * self.dt
            
        return q, False
