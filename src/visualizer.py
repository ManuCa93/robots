import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import datetime
import gc # Garbage collector 

class Visualizer:
    """Handles visualization, plotting, and saving animations."""
    
    @staticmethod
    def plot_end_effector_paths(history_data, title="End-Effector Trajectories", save_to_file=True):
        """
        Generates an AUTOMATIC ANIMATION synchronized in Real-Time.
        Saves both a GIF animation and a static PNG of the error plot.
        """
        if not history_data:
            print("[WARN] No history data to plot.")
            return

        if save_to_file:
            plt.switch_backend('Agg')

        print(f"[PLOT] Starting Processing for {len(history_data)} objects.")
        
        # Determine control method 
        title_lower = title.lower()
        if "joint" in title_lower:
            title_color = 'darkorange'
            mode_suffix = "joint"
        elif "rrmc" in title_lower:
            title_color = 'purple'
            mode_suffix = "rrmc"
        else:
            title_color = 'darkgreen'
            mode_suffix = "trajectory"
        
        if save_to_file:
            output_dir = "cubes_plot"
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Saving output to folder: {output_dir}")

        for name, data in history_data.items():
            
            # prepare raw data
            raw_pos = np.array(data['positions'])
            raw_times = np.array(data['times'])
            raw_errors = np.array(data.get('errors', [0]*len(raw_pos)))
            
            if len(raw_pos) < 2: continue
            
            # relative time
            t_start = raw_times[0]
            raw_t_rel = raw_times - t_start
            total_duration = raw_t_rel[-1]
            
            # resampling
            # this ensures uniform time steps for smooth animation for both rrmc and joint-space
            TARGET_FPS = 20
            
            num_frames = int(total_duration * TARGET_FPS)
            if num_frames < 2: num_frames = 2
            
            # New uniform time 
            t_rel = np.linspace(0, total_duration, num_frames)
            
            # interpolate positions and errors
            pos_x = np.interp(t_rel, raw_t_rel, raw_pos[:, 0])
            pos_y = np.interp(t_rel, raw_t_rel, raw_pos[:, 1])
            pos_z = np.interp(t_rel, raw_t_rel, raw_pos[:, 2])
            errors = np.interp(t_rel, raw_t_rel, raw_errors)
            
            positions = np.column_stack((pos_x, pos_y, pos_z))
            num_points = len(positions)
            
            # Generate filenames
            clean_name = name.replace(" ", "_")
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            filename_base = f"{clean_name}_{mode_suffix}_{timestamp}"


            # Save static error plot
            if save_to_file:
                fig_static = plt.figure(figsize=(10, 6))
                plt.plot(t_rel, errors, color='crimson', linewidth=2, label='Position Error')
                plt.fill_between(t_rel, errors, color='crimson', alpha=0.1)
                
                plt.title(f"Tracking Error Profile - {name.upper()} ({mode_suffix})", fontsize=14, color=title_color)
                plt.xlabel("Time (s)")
                plt.ylabel("Error (m)")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Statistiche nel grafico
                max_err = np.max(errors)
                avg_err = np.mean(errors)
                plt.text(0.02, 0.95, f'Max Error: {max_err:.4f} m\nAvg Error: {avg_err:.4f} m', 
                         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

                png_path = os.path.join(output_dir, f"{filename_base}_error_plot.png")
                plt.savefig(png_path)
                print(f"   [IMG] Saved Error Plot: {png_path}")
                plt.close(fig_static)

            # GIF creation
            fig = plt.figure(figsize=(18, 6))
            full_title = f"{title}\nREAL-TIME PLAYBACK: {name.upper()} ({total_duration:.2f}s)"
            fig.suptitle(full_title, fontsize=14, fontweight='bold', color=title_color)

            # 3D VIEW 
            ax3d = fig.add_subplot(1, 3, 1, projection='3d')
            ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                      color='navy', alpha=0.5, linewidth=1.0, linestyle='--')
            
            ax3d.scatter(positions[0,0], positions[0,1], positions[0,2], c='green', marker='^', s=50, label='Start')
            min_z_idx = np.argmin(positions[:, 2])
            ax3d.scatter(positions[min_z_idx, 0], positions[min_z_idx, 1], positions[min_z_idx, 2], 
                         c='orange', marker='x', s=50, label='Lowest Point')

            head_point, = ax3d.plot([], [], [], marker='o', color='red', markersize=10, linestyle='None')
            trail_line, = ax3d.plot([], [], [], color='red', alpha=0.6, linewidth=2)
            
            ax3d.set_title(f"3D Trajectory - {name}")
            ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
            ax3d.set_zlim(0, 0.7)
            ax3d.legend(loc='upper left')

            # Height profile 
            ax_z = fig.add_subplot(1, 3, 2)
            ax_z.plot(t_rel, positions[:, 2], color='purple', linewidth=2, alpha=0.3)
            vline_z = ax_z.axvline(t_rel[0], color='red', linewidth=2, alpha=0.8)
            point_z, = ax_z.plot([], [], marker='o', color='purple', markersize=8)
            
            ax_z.set_title("Height Profile (Z)")
            ax_z.set_xlabel("Time (s)"); ax_z.set_ylabel("Height (m)")
            ax_z.grid(True)

            # Error profile
            ax_err = fig.add_subplot(1, 3, 3)
            ax_err.plot(t_rel, errors, color='crimson', linewidth=1.5, alpha=0.4)
            vline_err = ax_err.axvline(t_rel[0], color='black', linewidth=1.5, linestyle='--')
            point_err, = ax_err.plot([], [], marker='o', color='crimson', markersize=8)
            
            ax_err.set_title("Position Error")
            ax_err.set_xlabel("Time (s)"); ax_err.set_ylabel("Error (m)")
            ax_err.grid(True)

            def update(frame):
                # 3D
                x, y, z = positions[frame]
                head_point.set_data([x], [y]); head_point.set_3d_properties([z])
                
                start = max(0, frame - 30)
                trail_line.set_data(positions[start:frame+1, 0], positions[start:frame+1, 1])
                trail_line.set_3d_properties(positions[start:frame+1, 2])

                # 2D Plots
                t = t_rel[frame]
                
                # Z Height
                vline_z.set_xdata([t])
                point_z.set_data([t], [z])
                
                # Error
                e = errors[frame]
                vline_err.set_xdata([t])
                point_err.set_data([t], [e])

                return head_point, trail_line, vline_z, point_z, vline_err, point_err

            interval_ms = int(1000 / TARGET_FPS)
            ani = FuncAnimation(fig, update, frames=num_points, interval=interval_ms, blit=False, repeat=True)

            if save_to_file:
                gif_path = os.path.join(output_dir, f"{filename_base}.gif")
                print(f"   [GIF] Saving Animation: {gif_path}")
                
                writer = PillowWriter(fps=TARGET_FPS)
                ani.save(gif_path, writer=writer)
                
                print(f"   [DONE] Saved GIF")
                
                plt.close(fig) 
                plt.clf()
                gc.collect()
            else:
                plt.show()

    @staticmethod
    def plot_joint_angles(history_data, title="Joint Angles Over Time", save_to_file=True):
        """
        Plot joint angles over time for all cubes.
        
        Args:
            history_data: Dictionary with cube trajectories containing joint_angles
            title: Plot title
            save_to_file: Whether to save to file
        """
        if not history_data:
            print("[WARN] No history data to plot.")
            return

        if save_to_file:
            plt.switch_backend('Agg')
            output_dir = "cubes_plot"
            os.makedirs(output_dir, exist_ok=True)

        print(f"[PLOT] Creating joint angle plots for {len(history_data)} objects.")
        
        # Determine control method from title
        title_lower = title.lower()
        if "joint" in title_lower:
            title_color = 'darkorange'
            mode_suffix = "joint"
        elif "rrmc" in title_lower:
            title_color = 'purple'
            mode_suffix = "rrmc"
        else:
            title_color = 'darkgreen'
            mode_suffix = "trajectory"

        for name, data in history_data.items():
            joint_angles_list = data.get('joint_angles', [])
            
            # Filter out None values
            valid_indices = [i for i, ja in enumerate(joint_angles_list) if ja is not None]
            if len(valid_indices) < 2:
                print(f"[WARN] Not enough joint angle data for {name}, skipping.")
                continue
            
            # Extract valid data
            joint_angles = np.array([joint_angles_list[i] for i in valid_indices])
            times = np.array([data['times'][i] for i in valid_indices])
            
            # Relative time
            t_rel = times - times[0]
            
            # Number of joints
            n_joints = joint_angles.shape[1]
            
            # Create figure with subplots for each joint
            fig, axes = plt.subplots(n_joints, 1, figsize=(12, 2*n_joints), sharex=True)
            fig.suptitle(f"{title} - {name.upper()}", fontsize=14, fontweight='bold', color=title_color)
            
            # Joint colors
            colors = plt.cm.tab10(np.linspace(0, 1, n_joints))
            
            for i in range(n_joints):
                ax = axes[i] if n_joints > 1 else axes
                ax.plot(t_rel, np.rad2deg(joint_angles[:, i]), color=colors[i], linewidth=2, label=f'Joint {i+1}')
                ax.set_ylabel(f'Joint {i+1}\n(deg)', fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(loc='upper right')
                
                # Add horizontal line at 0
                ax.axhline(0, color='black', linestyle=':', alpha=0.3)
            
            # Set x-label only on the bottom subplot
            if n_joints > 1:
                axes[-1].set_xlabel('Time (s)', fontsize=11)
            else:
                axes.set_xlabel('Time (s)', fontsize=11)
            
            plt.tight_layout()
            
            if save_to_file:
                clean_name = name.replace(" ", "_")
                timestamp = datetime.datetime.now().strftime("%H%M%S")
                filename = f"{clean_name}_{mode_suffix}_{timestamp}_joint_angles.png"
                png_path = os.path.join(output_dir, filename)
                plt.savefig(png_path, dpi=150)
                print(f"   [IMG] Saved Joint Angles Plot: {png_path}")
                plt.close(fig)
                gc.collect()
            else:
                plt.show()

    @staticmethod
    def plot_workspace_overview(object_manager):
        """Plot an overview of the workspace."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for name, pos in object_manager.cube_positions.items():
            ax.scatter(pos[0], pos[1], pos[2], marker='s', s=200, label=f'Cube {name}')
        ax.set_title("Workspace Overview")
        ax.legend()
        plt.show()