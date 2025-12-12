"""Data recording for trajectory analysis"""

import time
import numpy as np

class DataRecorder:
    """Records robot end-effector positions, timestamps, tracking errors, and joint angles."""
    
    def __init__(self):
        self.history = {}
        self.start_time = time.time()

    def log_pose(self, name, robot_pose, error_val=0.0, joint_angles=None):
        """Records the current End-Effector position, error, and joint angles."""

        if name is None: 
            return
        
        # Extract translation (x, y, z) directly assuming SE3 object
        pos = robot_pose.t
            
        # Calculate relative time from start
        current_t = time.time() - self.start_time
        
        # Initialize list for this cube if it doesn't exist
        if name not in self.history:
            self.history[name] = {
                'positions': [],
                'times': [],
                'errors': [],  
                'joint_angles': []  
            }
            
        # Save data
        self.history[name]['positions'].append(pos)
        self.history[name]['times'].append(current_t)
        self.history[name]['errors'].append(error_val) 
        
        # Save joint angles if provided
        if joint_angles is not None:
            self.history[name]['joint_angles'].append(np.array(joint_angles).copy())
        else:
            # Append None to keep lists synchronized
            self.history[name]['joint_angles'].append(None)

    def get_data(self):
        """Returns the entire recorded history."""
        
        return self.history