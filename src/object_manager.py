"""Object and Plate Management"""

import numpy as np
import random
from spatialmath import SE3
from spatialgeometry import Cuboid


class ObjectManager:
    """Manages cubes and plates in the environment."""
    
    def __init__(self, cube_size=0.06, plate_size=0.1, plate_height=0.005):
        """
        Initialize the object manager.
        
        Args:
            cube_size: Size of the cubes
            plate_size: Size of the plates
            plate_height: Height of the plates
        """
        self.cube_size = cube_size
        self.plate_size = plate_size
        self.plate_height = plate_height
        
        self.cube_center_z = cube_size / 2
        self.plate_center_z = plate_height / 2
        self.pick_z = self.cube_center_z + cube_size / 2
        
        self.cubes = {}
        self.plates = {}
        self.cube_positions = {}
        self.plate_positions = {}
        
    def create_cubes(self, positions, env):
        """
        Create cube objects at specified positions.
        
        Args:
            positions: Dictionary mapping color names to (x, y, z) positions
            env: RobotEnvironment instance
        """
        colors = {
            'red': [1, 0, 0, 0.8],
            'blue': [0, 0, 1, 0.8],
            'green': [0, 1, 0, 0.8],
            'yellow': [1, 1, 0, 0.8]
        }
        
        for name, pos in positions.items():
            cube = Cuboid(
                [self.cube_size, self.cube_size, self.cube_size],
                pose=SE3(pos[0], pos[1], self.cube_center_z),
                color=colors.get(name, [0.5, 0.5, 0.5, 0.8])
            )
            self.cubes[name] = cube
            self.cube_positions[name] = pos
            env.add_object(cube)
            
        print(f"[INFO] Created {len(self.cubes)} cubes")
        
    def generate_plate_positions(self, terrain_bounds, num_plates=4):
        """
        Generate random non-overlapping positions for plates.
        
        Args:
            terrain_bounds: Dictionary with x_min, x_max, y_min, y_max
            num_plates: Number of plates to place
            
        Returns:
            List of positions
        """
        min_distance = self.plate_size * 1.5
        positions = []
        attempts = 0
        max_attempts = 100
        
        while len(positions) < num_plates and attempts < max_attempts:
            new_pos = np.array([
                random.uniform(terrain_bounds['x_min'], terrain_bounds['x_max']),
                random.uniform(terrain_bounds['y_min'], terrain_bounds['y_max']),
                self.plate_center_z
            ])
            
            if (self._is_within_bounds(new_pos, terrain_bounds) and
                all(np.linalg.norm(new_pos[:2] - pos[:2]) >= min_distance for pos in positions)):
                positions.append(new_pos)
            attempts += 1
            
        if len(positions) < num_plates:
            raise ValueError("Could not generate enough unique positions")
            
        return positions
    
    def create_plates(self, terrain_bounds, env):
        """
        Create plate objects at random positions.
        
        Args:
            terrain_bounds: Dictionary with x_min, x_max, y_min, y_max
            env: RobotEnvironment instance
        """
        colors = {
            'red': [1, 0, 0, 0.5],
            'blue': [0, 0, 1, 0.5],
            'green': [0, 1, 0, 0.5],
            'yellow': [1, 1, 0, 0.5]
        }
        
        plate_positions = self.generate_plate_positions(terrain_bounds, num_plates=4)
        
        for idx, (name, color) in enumerate(colors.items()):
            pos = plate_positions[idx]
            plate = Cuboid(
                [self.plate_size, self.plate_size, self.plate_height],
                pose=SE3(*pos),
                color=color
            )
            self.plates[name] = plate
            self.plate_positions[name] = pos
            env.add_object(plate)
            
        print(f"[INFO] Created {len(self.plates)} plates with random positions")
        
    def _is_within_bounds(self, position, bounds):
        """Check if position is within bounds."""
        x, y = position[:2]
        return (
            bounds['x_min'] + self.plate_size / 2 <= x <= bounds['x_max'] - self.plate_size / 2 and
            bounds['y_min'] + self.plate_size / 2 <= y <= bounds['y_max'] - self.plate_size / 2
        )
        
    def get_pick_place_pairs(self):
        """
        Get pick and place positions for each object.
        
        Returns:
            Dictionary mapping color names to (pick_pos, place_pos) tuples
        """
        pairs = {}
        for name in self.cube_positions.keys():
            pick_pos = self.cube_positions[name]
            place_pos = self.plate_positions[name].copy()
            place_pos[2] = max(self.plate_center_z + self.cube_size, place_pos[2])
            pairs[name] = (pick_pos, place_pos)
        return pairs
