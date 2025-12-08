"""Object and Plate Management"""

import numpy as np
import random
import threading
import time
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
        self.buckets_positions = {}
        self.object_place_positions = {}
        
        # Circular motion parameters
        self.motion_enabled = False
        self.motion_thread = None
        self.motion_lock = threading.Lock()
        self.circle_center = np.array([0.425, 0.0])  # Center of workspace
        self.circle_radius = 0.15  # Radius of circular motion
        self.angular_velocity = 0.3  # rad/s (slower cubes, easier to catch)
        self.motion_start_time = None
        self.initial_angles = {}  # Store initial angle for each cube
        self.picked_cubes = set()  # Track cubes that are picked up (stop their motion)
        
    def create_cubes(self, positions, env):
        """
        Create cube objects at specified positions.
        
        Args:
            positions: Dictionary mapping color names to (x, y, z) positions
            env: RobotEnvironment instance
        """
        for name, pos in positions.items():
            base_color = self.get_base_color(name)
            colors = {
                'red': [1, 0, 0, 0.8],
                'blue': [0, 0, 1, 0.8],
                'green': [0, 1, 0, 0.8],
                'yellow': [1, 1, 0, 0.8]
            }
            cube = Cuboid(
                [self.cube_size, self.cube_size, self.cube_size],
                pose=SE3(pos[0], pos[1], self.cube_center_z),
                color=colors.get(base_color, [0.5, 0.5, 0.5, 0.8])
            )
            self.cubes[name] = cube
            self.cube_positions[name] = pos
            env.add_object(cube)
        
        print(self.cube_positions)
        
        # Initialize cube angles for circular motion
        num_cubes = len(self.cubes)
        for idx, name in enumerate(self.cubes.keys()):
            # Distribute cubes evenly around the circle
            self.initial_angles[name] = (2 * np.pi * idx) / num_cubes
            
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
    
    def get_base_color(self, name):
        """Extract base color from cube name (e.g. 'red1' -> 'red')."""
        if name.startswith('red'):
            return 'red'
        if name.startswith('green'):
            return 'green'
        if name.startswith('yellow'):
            return 'yellow'
        if name.startswith('blue'):
            return 'blue'
        return name
        
    def get_pick_place_pairs(self):
        """
        Get pick and place positions for each object.
        
        Returns:
            Dictionary mapping color names to (pick_pos, place_pos) tuples
        """
        pairs = {}
        
        # Each bucket can contain up to 9 cubes in a 3x3 grid
        grid_size = 3
        slot_spacing = self.cube_size * 1.5
        placed_counts = {color: 0 for color in self.buckets_positions.keys()}
        offset_base = (grid_size - 1) / 2

        self.object_place_positions = {}

        for name, pick_pos in self.cube_positions.items():
            base_color = self.get_base_color(name)
            if base_color not in self.buckets_positions:
                # Skip objects whose color has no associated bucket
                continue

            count = placed_counts.get(base_color, 0)
            max_slots = grid_size ** 2
            if count >= max_slots:
                print(f"[WARN] Bucket '{base_color}' is full (max {max_slots} cubes). Skipping {name}.")
                continue

            # Center position of the bucket for this color
            bucket_center = self.buckets_positions[base_color].copy()

            # Compute 3x3 grid offsets inside the bucket
            row = count // grid_size
            col = count % grid_size
            offset_x = (col - offset_base) * slot_spacing
            offset_y = (row - offset_base) * slot_spacing

            place_pos = bucket_center.copy()
            place_pos[0] += offset_x
            place_pos[1] += offset_y

            # Ensure Z is on the bucket floor (bucket_center.z already encodes drop height)
            place_pos[2] = bucket_center[2]

            placed_counts[base_color] = count + 1

            self.object_place_positions[name] = place_pos.copy()
            pairs[name] = (pick_pos, place_pos)

        return pairs
    
    def start_circular_motion(self, env):
        """Start circular motion of cubes in a separate thread."""
        if self.motion_enabled:
            print("[INFO] Circular motion already running")
            return
        
        self.motion_enabled = True
        self.motion_start_time = time.time()
        self.motion_thread = threading.Thread(target=self._circular_motion_loop, args=(env,), daemon=True)
        self.motion_thread.start()
        print(f"[INFO] Started circular motion: radius={self.circle_radius}m, omega={self.angular_velocity}rad/s")
    
    def stop_circular_motion(self):
        """Stop circular motion of cubes."""
        self.motion_enabled = False
        if self.motion_thread:
            self.motion_thread.join(timeout=1.0)
        print("[INFO] Stopped circular motion")
    
    def _circular_motion_loop(self, env):
        """Background thread that updates cube positions in a circle."""
        dt = 0.02  # Update every 20ms
        
        while self.motion_enabled:
            current_time = time.time() - self.motion_start_time
            
            with self.motion_lock:
                for name, cube in self.cubes.items():
                    if name not in self.initial_angles:
                        continue
                    
                    # Skip cubes that are picked up or released
                    if name in self.picked_cubes:
                        continue
                    
                    # Calculate current angle
                    angle = self.initial_angles[name] + self.angular_velocity * current_time
                    
                    # Calculate position on circle
                    x = self.circle_center[0] + self.circle_radius * np.cos(angle)
                    y = self.circle_center[1] + self.circle_radius * np.sin(angle)
                    z = self.cube_center_z
                    
                    # Update cube position (both visual and stored)
                    cube.T = SE3(x, y, z)
                    self.cube_positions[name] = np.array([x, y, self.pick_z])
            
            time.sleep(dt)
    
    def get_current_cube_position(self, name):
        """Get current position of a cube (thread-safe)."""
        with self.motion_lock:
            if name in self.cube_positions:
                return self.cube_positions[name].copy()
            return None
    
    def mark_cube_picked(self, name):
        """Mark a cube as picked (stops its circular motion)."""
        with self.motion_lock:
            self.picked_cubes.add(name)
            print(f"[MOTION] Cube {name} marked as picked, motion stopped")
    
    def mark_cube_released(self, name):
        """Mark a cube as released (it stays where placed, doesn't resume motion)."""
        with self.motion_lock:
            # Keep cube in picked_cubes set so motion loop never touches it again
            if name not in self.picked_cubes:
                self.picked_cubes.add(name)
            print(f"[MOTION] Cube {name} marked as released (motion disabled permanently)")