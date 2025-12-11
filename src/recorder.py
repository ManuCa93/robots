"""Data recording for trajectory analysis"""

import time
import numpy as np

class DataRecorder:
    """Records robot end-effector positions, timestamps, and tracking errors."""
    
    def __init__(self):
        # Structure: { 'cube_name': { 'positions': [...], 'times': [...], 'errors': [...] } }
        self.history = {}
        self.start_time = time.time()

    def log_pose(self, name, robot_pose, error_val=0.0):
        """
        Records the current End-Effector position and error.
        
        Args:
            name: Name of the cube being manipulated (e.g., 'red1')
            robot_pose: SE3 pose (must have .t attribute)
            error_val: Current tracking error (default 0.0 for Joint Space)
        """
        if name is None: 
            return
        
        # Extract translation (x, y, z) directly assuming SE3 object
        # (We removed the if/else checks as requested, assuming input is always SE3)
        pos = robot_pose.t
            
        # Calculate relative time from start
        current_t = time.time() - self.start_time
        
        # Initialize list for this cube if it doesn't exist
        if name not in self.history:
            self.history[name] = {
                'positions': [],
                'times': [],
                'errors': []  # <--- NEW: List to store errors
            }
            
        # Save data
        self.history[name]['positions'].append(pos)
        self.history[name]['times'].append(current_t)
        self.history[name]['errors'].append(error_val) # <--- NEW: Save the error value

    def get_data(self):
        """Returns the entire recorded history."""
        return self.history