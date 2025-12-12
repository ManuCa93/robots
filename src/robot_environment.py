"""Robot Environment Management"""

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.backends.swift import Swift
from spatialmath import SE3
from spatialgeometry import Cuboid


class RobotEnvironment:
    """Manages the Swift environment, robot, and terrain."""
    
    def __init__(self, terrain_bounds=None):
        """Initialize the robot environment."""
        
        self.env = Swift()
        self.panda = rtb.models.Panda()
        
        self.bucket_positions = [
            (0.30, -0.50),  
            (0.55, -0.50),  
            (0.30, 0.50),   
            (0.55, 0.50),   
        ]

        # Extract X and Y coordinates
        self.bucket_x = [pos[0] for pos in self.bucket_positions]
        self.bucket_y = [pos[1] for pos in self.bucket_positions]

        # Calculate bounds with margin
        margin = 0.1
        self.terrain_bounds = {
            "x_min": min(self.bucket_x) - margin,  # 0.30 - 0.1 = 0.20
            "x_max": max(self.bucket_x) + margin,  # 0.55 + 0.1 = 0.65
            "y_min": min(self.bucket_y) - margin,  # -0.50 - 0.1 = -0.60
            "y_max": max(self.bucket_y) + margin,  # 0.50 + 0.1 = 0.60
        }
        
        # Home configuration
        self.q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        
        self.terrain = None
        
    def launch(self, realtime=True):
        """Launch the Swift environment."""
        self.env.launch(realtime=realtime, comms="rtc")
        self.env.add(self.panda)
        self._create_terrain()
        print("[INFO] Environment launched successfully")
        
    def _create_terrain(self):
        """Create and add terrain to the environment."""
        self.terrain = Cuboid(
            [self.terrain_bounds['x_max'] - self.terrain_bounds['x_min'], 
            self.terrain_bounds['y_max'] - self.terrain_bounds['y_min'], 
            0.001],  
            pose=SE3((self.terrain_bounds['x_min'] + self.terrain_bounds['x_max']) / 2, 
                     (self.terrain_bounds['y_min'] + self.terrain_bounds['y_max']) / 2, 
                     0),
            color=[0.5, 0.5, 0.5, 0.3]
        )
        self.env.add(self.terrain)
        
    def add_object(self, obj):
        """Add a geometric object to the environment."""
        self.env.add(obj)
        
    def step(self, dt=0.01):
        """Step the environment forward."""
        self.env.step(dt)
        
    def set_robot_config(self, q):
        """Set the robot joint configuration."""
        self.panda.q = q