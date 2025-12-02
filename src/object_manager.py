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
        
        print(self.cube_positions)
            
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
    
    def create_buckets(self, terrain_bounds, env, cube_height):
        """
        Create plate objects at random positions.
        
        Args:
            terrain_bounds: Dictionary with x_min, x_max, y_min, y_max
            env: RobotEnvironment instance
        """
        # colors = {
        #     'red': [1, 0, 0, 0.5],
        #     'blue': [0, 0, 1, 0.5],
        #     'green': [0, 1, 0, 0.5],
        #     'yellow': [1, 1, 0, 0.5]
        # }
        
        # # plate_positions = self.generate_plate_positions(terrain_bounds, num_plates=4)
        
        # for idx, (name, color) in enumerate(colors.items()):
        #     pos = plate_positions[idx]
        #     plate = Cuboid(
        #         [self.plate_size, self.plate_size, self.plate_height],
        #         pose=SE3(*pos),
        #         color=color
        #     )
        #     self.plates[name] = plate
        #     self.plate_positions[name] = pos
        #     env.add_object(plate)
            
        # print(f"[INFO] Created {len(self.plates)} plates with random positions")
        color_names = ['red', 'blue', 'green', 'yellow']
        color_map = {
            'red': [1, 0, 0, 0.5],
            'blue': [0, 0, 1, 0.5],
            'green': [0, 1, 0, 0.5],
            'yellow': [1, 1, 0, 0.5]
        }

        # Posizioni fisse
        fixed_positions = [
            (0.30, -0.50),  # bottom-left
            (0.55, -0.50),  # bottom-right
            (0.30, 0.50),   # top-left
            (0.55, 0.50),   # top-right
        ]

        # Randomizza l'assegnazione colori -> posizioni
        import random
        shuffled_colors = color_names.copy()
        random.shuffle(shuffled_colors)

        corners = {shuffled_colors[i]: fixed_positions[i] for i in range(4)}

        # Calcola terrain_bounds in base alle posizioni dei bucket
        bucket_positions = list(corners.values())
        bucket_x = [pos[0] for pos in bucket_positions]
        bucket_y = [pos[1] for pos in bucket_positions]

        margin = 0.1
        terrain_bounds = {
            "x_min": min(bucket_x) - margin,
            "x_max": max(bucket_x) + margin,
            "y_min": min(bucket_y) - margin,
            "y_max": max(bucket_y) + margin,
        }

        print(f"[INFO] Terrain bounds (from bucket positions):")
        print(f"  X: [{terrain_bounds['x_min']:.2f}, {terrain_bounds['x_max']:.2f}]")
        print(f"  Y: [{terrain_bounds['y_min']:.2f}, {terrain_bounds['y_max']:.2f}]")

        print(f"\n[INFO] Fixed bucket positions (randomized assignment):")
        for color, (x, y) in corners.items():
            print(f"  {color.upper():7s}: ({x:.2f}, {y:.2f})")

        # Parametri del secchio
        bucket_inner_diameter = 0.12
        bucket_wall_thickness = 0.01
        bucket_outer_diameter = bucket_inner_diameter + 2 * bucket_wall_thickness
        bucket_base_thickness = 0.01
        bucket_wall_height = 0.08
        bucket_drop_height = bucket_base_thickness + cube_height / 2
        bucket_wall_center_z = bucket_base_thickness + bucket_wall_height / 2

        self.buckets_positions = {}
        bucket_objects = {}

        print(f"\n[ASSIGNMENT] Creating buckets at fixed positions:")

        for color, (corner_x, corner_y) in corners.items():
            bucket_pos = np.array([corner_x, corner_y, bucket_drop_height])
            self.buckets_positions[color] = bucket_pos

            print(f"  {color.upper():7s} at ({corner_x:.2f}, {corner_y:.2f})")

            bucket_color = color_map[color]
            base = Cuboid(
                [bucket_outer_diameter, bucket_outer_diameter, bucket_base_thickness],
                pose=SE3(corner_x, corner_y, bucket_base_thickness / 2),
                color=bucket_color
            )
            env.add_object(base)

            walls = []
            wall_offset = bucket_outer_diameter / 2 - bucket_wall_thickness / 2
            wall_specs = [
                ([bucket_outer_diameter, bucket_wall_thickness, bucket_wall_height], (0, wall_offset)),
                ([bucket_outer_diameter, bucket_wall_thickness, bucket_wall_height], (0, -wall_offset)),
                ([bucket_wall_thickness, bucket_outer_diameter, bucket_wall_height], (wall_offset, 0)),
                ([bucket_wall_thickness, bucket_outer_diameter, bucket_wall_height], (-wall_offset, 0)),
            ]

            for dims, (dx, dy) in wall_specs:
                wall = Cuboid(
                    dims,
                    pose=SE3(corner_x + dx, corner_y + dy, bucket_wall_center_z),
                    color=bucket_color
                )
                env.add_object(wall)
                walls.append(wall)

            bucket_objects[color] = {
                "base": base,
                "walls": walls
            }

        print(f"\n[OK] Fixed buckets placed")
        
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
            bucket_pos = self.buckets_positions[name].copy()
            bucket_pos[2] = max(self.plate_center_z + self.cube_size, bucket_pos[2])
            pairs[name] = (pick_pos, bucket_pos)
        print(pairs["red"])
        return pairs