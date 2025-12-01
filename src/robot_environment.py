"""Robot Environment Management"""

import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox.backends.swift import Swift
from spatialmath import SE3
from spatialgeometry import Cuboid


class RobotEnvironment:
    """Manages the Swift environment, robot, and terrain."""
    
    def __init__(self, terrain_bounds=None):
        """
        Initialize the robot environment.
        
        Args:
            terrain_bounds: Dictionary with x_min, x_max, y_min, y_max
        """
        self.env = Swift()
        self.panda = rtb.models.Panda()
        
        # Default terrain bounds
        if terrain_bounds is None:
            terrain_bounds = {
                "x_min": 0.2, "x_max": 0.6,
                "y_min": -0.8, "y_max": 0.8
            }
        self.terrain_bounds = terrain_bounds
        
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
        bounds = self.terrain_bounds
        self.terrain = Cuboid(
            [bounds['x_max'] - bounds['x_min'], 
             bounds['y_max'] - bounds['y_min'], 
             0.001],
            pose=SE3((bounds['x_min'] + bounds['x_max']) / 2, 
                     (bounds['y_min'] + bounds['y_max']) / 2, 
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
