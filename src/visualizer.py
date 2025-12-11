import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class Visualizer:
    """Handles visualization and plotting."""
    
    @staticmethod
    def plot_end_effector_paths(history_data, title="End-Effector Trajectories"):
        """
        Generates an AUTOMATIC ANIMATION of the path.
        Shows the robot moving in 3D with a trail effect.
        """
        if not history_data:
            print("[WARN] No history data to plot.")
            return

        print(f"[PLOT] Starting Auto-Animation for {len(history_data)} objects.")
        print(">>> CLOSE THE WINDOW TO PROCEED TO THE NEXT OBJECT <<<")
        
        # Iteriamo su ogni oggetto manipolato
        for name, data in history_data.items():
            positions = np.array(data['positions'])
            times = np.array(data['times'])
            num_points = len(positions)

            if num_points < 2: 
                continue
            
            t_rel = times - times[0]
            
            # --- SETUP FINESTRA ---
            fig = plt.figure(figsize=(14, 7))
            full_title = f"{title}\nPLAYBACK: {name.upper()} (Home -> Pick -> Place -> Home)"
            fig.suptitle(full_title, fontsize=14, fontweight='bold', color='darkgreen')

            # -----------------------------
            # 1. 3D ANIMATION VIEW
            # -----------------------------
            ax3d = fig.add_subplot(1, 2, 1, projection='3d')
            
            # Sfondo: traiettoria completa in grigio chiaro (riferimento)
            ax3d.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                      color='grey', alpha=0.5, linewidth=0.8, linestyle='--')

            # Elementi statici (Home, Pick point)
            ax3d.scatter(positions[0,0], positions[0,1], positions[0,2], c='green', marker='^', s=50, label='HOME')
            min_z_idx = np.argmin(positions[:, 2])
            ax3d.scatter(positions[min_z_idx, 0], positions[min_z_idx, 1], positions[min_z_idx, 2], 
                         c='orange', marker='x', s=50, label='Lowest Point')

            # ELEMENTI DINAMICI (Quelli che si muoveranno)
            # 1. Il punto corrente (Testa del robot)
            head_point, = ax3d.plot([], [], [], marker='o', color='red', markersize=10, linestyle='None', label='End Effector')
            # 2. La scia (Gli ultimi N punti)
            trail_line, = ax3d.plot([], [], [], color='red', alpha=0.6, linewidth=2)

            ax3d.set_title(f"Live Trajectory - {name}")
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.set_zlim(0, 0.7)
            ax3d.legend(loc='upper left')

            # -----------------------------
            # 2. 2D HEIGHT PROFILE (Sync)
            # -----------------------------
            ax_z = fig.add_subplot(1, 2, 2)
            ax_z.plot(t_rel, positions[:, 2], color='purple', linewidth=2, alpha=0.3) # Linea base sbiadita
            
            # Linea verticale che scorre
            vline = ax_z.axvline(t_rel[0], color='red', linewidth=2, alpha=0.8)
            # Punto che scorre sulla curva
            z_point, = ax_z.plot([], [], marker='o', color='purple', markersize=8)

            ax_z.set_title("Height Synchronization")
            ax_z.set_xlabel("Time (s)")
            ax_z.set_ylabel("Height Z (m)")
            ax_z.grid(True)

            # --- FUNZIONE DI AGGIORNAMENTO ANIMAZIONE ---
            def update(frame):
                # 'frame' è l'indice attuale (da 0 a num_points)
                
                # 1. Aggiorna posizione 3D (Testa)
                x, y, z = positions[frame]
                head_point.set_data([x], [y])     # x, y devono essere liste
                head_point.set_3d_properties([z]) # z a parte per il 3D

                # 2. Aggiorna la scia (ultimi 20 punti)
                start_trail = max(0, frame - 20)
                trail_x = positions[start_trail:frame+1, 0]
                trail_y = positions[start_trail:frame+1, 1]
                trail_z = positions[start_trail:frame+1, 2]
                
                trail_line.set_data(trail_x, trail_y)
                trail_line.set_3d_properties(trail_z)

                # 3. Aggiorna grafico 2D (Linea verticale e punto)
                vline.set_xdata([t_rel[frame]])
                z_point.set_data([t_rel[frame]], [z])

                return head_point, trail_line, vline, z_point

            # Creazione Animazione
            # interval=20 -> 20ms tra un frame e l'altro (velocità)
            # repeat=True -> ricomincia da capo alla fine
            ani = FuncAnimation(fig, update, frames=num_points, interval=10, blit=False, repeat=True)

            plt.show()

    @staticmethod
    def plot_workspace_overview(object_manager):
        """Plot an overview of the workspace with cubes and plates."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for name, pos in object_manager.cube_positions.items():
            ax.scatter(pos[0], pos[1], pos[2], marker='s', s=200, label=f'Cube {name}')
        ax.set_title("Workspace Overview")
        ax.legend()
        plt.show()