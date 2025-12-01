"""Visualization and Plotting"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    """Handles visualization and plotting."""
    
    @staticmethod
    def plot_rrmc_waypoints(trajectories):
        """
        Plot target waypoints for RRMC motion for each object.
        
        Args:
            trajectories: Dictionary mapping names to trajectory data
        """
        for name, traj_data in trajectories.items():
            poses = traj_data["poses"]
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Extract positions from poses
            waypoint_names = ["home", "pick_above", "pick", "place_above", "place"]
            positions = []
            
            for wp_name in waypoint_names:
                if wp_name in poses:
                    pose = poses[wp_name]
                    positions.append(pose.t)
            
            positions = np.array(positions)
            
            # Plot waypoints
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                       c='red', marker='o', s=100, label='Waypoints')
            
            # Plot path connecting waypoints
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                    'b--', alpha=0.6, label='Path')
            
            # Annotate waypoints
            for i, wp_name in enumerate(waypoint_names):
                if i < len(positions):
                    ax.text(positions[i, 0], positions[i, 1], positions[i, 2], 
                            wp_name, fontsize=8)
            
            ax.set_title(f"RRMC Target Waypoints for {name}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.legend()
            plt.show()
            
    @staticmethod
    def plot_workspace_overview(object_manager):
        """
        Plot an overview of the workspace with cubes and plates.
        
        Args:
            object_manager: ObjectManager instance
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot cubes
        for name, pos in object_manager.cube_positions.items():
            ax.scatter(pos[0], pos[1], pos[2], marker='s', s=200, label=f'Cube {name}')
        
        # Plot plates
        for name, pos in object_manager.plate_positions.items():
            ax.scatter(pos[0], pos[1], pos[2], marker='^', s=200, label=f'Plate {name}')
        
        ax.set_title("Workspace Overview")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend()
        plt.show()
