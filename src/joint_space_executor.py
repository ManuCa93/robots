"""Joint-Space Pick and Place Execution with Dynamic Tracking"""

import time
import numpy as np
from spatialmath import SE3

class JointSpaceExecutor:
    """Executes pick-and-place operations using dynamic joint-space servoing."""
    
    def __init__(self, robot, env, joint_space_controller, object_manager, recorder=None, sleep_dt=0.02, gravity_acceleration=2.0):
        self.robot = robot
        self.env = env
        self.controller = joint_space_controller
        self.object_manager = object_manager
        self.recorder = recorder  # Reference to DataRecorder
        self.sleep_dt = sleep_dt
        self.gravity_acceleration = gravity_acceleration
        
        # Guadagno del servoing
        self.servo_gain = 0.8 
        self.current_cube_name = None  # Track current object for logging

    def execute_all(self, trajectories):
        for idx, (name, traj_data) in enumerate(trajectories.items(), 1):
            print(f"\n[INFO] [{idx}/{len(trajectories)}] Executing Joint-Space P&P for {name}")
            self.execute_single(name, traj_data)

    def execute_single(self, name, trajectory_data):
        self.current_cube_name = name  # Set context for logging
        cube = self.object_manager.cubes[name]
        
        try:
            # --- FASE 0: Avvicinamento Statico Fluido ---
            # Ci portiamo in zona "alta" (25cm) con una traiettoria calcolata (spline)
            # Questo evita lo scatto iniziale dalla Home.
            print(f"[Joint-Space] {name}: Smooth Approach (High)...")
            cube_pos_init = self.object_manager.get_current_cube_position(name)
            
            if cube_pos_init is not None:
                # Target iniziale ALTO: 25cm sopra il cubo
                target_approach = SE3(cube_pos_init[0], cube_pos_init[1], cube_pos_init[2] + 0.25) * SE3.Rx(np.pi)
                
                sol = self.robot.ikine_LM(target_approach, q0=self.robot.q)
                if sol.success:
                    self._move_joints_direct(sol.q)
            
            # --- FASE 1: Discesa di Ingaggio (NUOVA) ---
            # Invece di saltare subito a 12cm, scendiamo fluidamente da 25cm a 12cm
            # MENTRE iniziamo a inseguire (XY). Questo elimina lo scatto.
            print(f"[Joint-Space] {name}: Swooping down to track...")
            self._visual_servo_descend_trajectory(name, start_offset=0.25, end_offset=0.12, duration=1.0)

            # --- FASE 2: Discesa di Presa ---
            # Ora siamo a 12cm, allineati e stabili. Scendiamo a prendere (1.8cm).
            print(f"[Joint-Space] {name}: Final Grasp Descent...")
            self._visual_servo_descend_trajectory(name, start_offset=0.12, end_offset=0.010, duration=1.0)
            
            # --- FASE 3: Presa (Attach) ---
            self.object_manager.mark_cube_picked(name)
            
            T_ee_contact = self.robot.fkine(self.robot.q)
            T_cube_contact = cube.T
            T_rel = T_ee_contact.inv() * T_cube_contact
            
            print(f"[Joint-Space] {name}: Attached. Lifting...")

            # --- FASE 4: Risalita (Lift) ---
            # Risaliamo a 20cm (coordinate Mondo)
            current_t = T_ee_contact.t
            lift_pose = SE3(current_t[0], current_t[1], current_t[2] + 0.20) * SE3.Rx(np.pi)
            
            self._move_to_pose_interpolated(lift_pose, cube, T_rel)
            
            # --- FASE 5: Trasporto al Bucket ---
            print(f"[Joint-Space] {name}: Transporting...")
            place_pos = self.object_manager.object_place_positions.get(name)
            
            if place_pos is not None:
                target_place_above = SE3(place_pos) * SE3(0, 0, 0.20) * SE3.Rx(np.pi)
                self._move_to_pose_interpolated(target_place_above, cube, T_rel)
            
            # --- FASE 6: Rilascio ---
            print(f"[Joint-Space] {name}: Dropping...")
            self.object_manager.mark_cube_released(name)
            
            if place_pos is not None:
                with self.object_manager.motion_lock:
                    self.object_manager.cube_positions[name] = place_pos.copy()
                
                cube_transform = cube.T
                current_cube_pos = cube_transform.t if hasattr(cube_transform, 't') else SE3(cube_transform).t
                self._simulate_gravity_drop(cube, current_cube_pos, place_pos)

            # --- FASE 7: Home ---
            print(f"[Joint-Space] {name}: Returning Home...")
            self._move_joints_direct(self.env.q_home)
            
            print(f"[Joint-Space] {name}: Done.")
            
        except Exception as e:
            print(f"[ERROR] Fail {name}: {e}")
            import traceback
            traceback.print_exc()

    def _visual_servo_descend_trajectory(self, name, start_offset, end_offset, duration):
        """
        Gestisce la discesa interpolata (Z) mentre corregge l'errore planare (XY) in tempo reale.
        Usata sia per l'avvicinamento (25->12cm) che per la presa (12->1.8cm).
        """
        steps = int(duration / self.sleep_dt)
        
        for i in range(steps):
            t = i / steps
            current_z_offset = start_offset * (1 - t) + end_offset * t
            
            cube_pos = self.object_manager.get_current_cube_position(name)
            if cube_pos is None: break
            
            # Calcolo target sicuro (impedisce Z < altezza cubo se siamo alla fine)
            target_z = cube_pos[2] + current_z_offset
            
            # Rotazione 180° su X (pinza in giù)
            target_pose = SE3(cube_pos[0], cube_pos[1], target_z) * SE3.Rx(np.pi)
            
            sol = self.robot.ikine_LM(target_pose, q0=self.robot.q)
            if sol.success:
                # Servo Gain alto (0.85) per tracking reattivo
                q_next = self.robot.q + 0.85 * (sol.q - self.robot.q)
                self._step_robot(q_next)
            
            time.sleep(self.sleep_dt)

    def _move_to_pose_interpolated(self, target_pose, cube=None, T_rel=None):
        sol = self.robot.ikine_LM(target_pose, q0=self.robot.q)
        if sol.success:
            traj = self.controller.generate_trajectory(self.robot.q, sol.q)
            for q in traj:
                self._step_robot(q, cube, T_rel, attached=(cube is not None))

    def _move_joints_direct(self, q_target):
        # Movimento spline fluido per lunghe distanze
        traj = self.controller.generate_trajectory(self.robot.q, q_target)
        for q in traj:
            self._step_robot(q)

    def _step_robot(self, q, cube=None, T_rel=None, attached=False):
        self.robot.q = q
        self.env.set_robot_config(q)
        
        # --- LOGGING ---
        if self.recorder and self.current_cube_name:
            current_pose = self.robot.fkine(q)
            self.recorder.log_pose(self.current_cube_name, current_pose)
        # ----------------
        
        if attached and cube is not None and T_rel is not None:
            T_ee = self.robot.fkine(q)
            cube.T = T_ee * T_rel
            
        self.env.step(0.01)

    def _simulate_gravity_drop(self, cube, start_pos, target_pos):
        drop_distance = start_pos[2] - target_pos[2]
        if drop_distance <= 0:
            cube.T = SE3(target_pos[0], target_pos[1], target_pos[2])
            return
        
        fall_time = np.sqrt(2 * drop_distance / self.gravity_acceleration)
        num_steps = max(int(fall_time / 0.015), 15) 
        
        for step in range(num_steps + 1):
            t = (step / num_steps) * fall_time
            current_z = start_pos[2] - 0.5 * self.gravity_acceleration * t**2
            current_z = max(current_z, target_pos[2])
            cube.T = SE3(target_pos[0], target_pos[1], current_z)
            self.env.step(0.01)
            time.sleep(0.015)
            
        cube.T = SE3(target_pos[0], target_pos[1], target_pos[2])