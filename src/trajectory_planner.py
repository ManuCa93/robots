"""Trajectory Planning and IK Computation"""

import numpy as np
from spatialmath import SE3


class TrajectoryPlanner:
    """Plans trajectories and computes inverse kinematics."""
    
    def __init__(self, robot, q_home, approach_height=0.12, joint_space_controller=None):
        """
        Initialize trajectory planner.
        
        Args:
            robot: Robot model
            q_home: Home configuration
            approach_height: Height above object for approach
            joint_space_controllerma : Optional JointSpaceController for generating joint trajectories
        """
        self.robot = robot
        self.q_home = q_home
        self.approach_height = approach_height
        self.joint_space_controller = joint_space_controller
        
    def compute_ik(self, position, z_offset=0.0):
        """Compute inverse kinematics for a position."""

        target = np.array(position).copy()
        target[2] += z_offset
        
        pose = SE3(target[0], target[1], target[2]) * SE3.Rx(np.pi)
        result = self.robot.ikine_LM(pose, q0=self.q_home)
        
        return result.q, result.success
        
    def plan_pick_and_place(self, pick_pos, place_pos):
        """Plan a complete pick-and-place trajectory."""

        # Create SE3 poses
        poses = {
            "home": self.robot.fkine(self.q_home),
            "pick_above": SE3(pick_pos[0], pick_pos[1], pick_pos[2] + self.approach_height) * SE3.Rx(np.pi),
            "pick": SE3(pick_pos[0], pick_pos[1], pick_pos[2]) * SE3.Rx(np.pi),
            "place_above": SE3(place_pos[0], place_pos[1], place_pos[2] + self.approach_height) * SE3.Rx(np.pi),
            "place": SE3(place_pos[0], place_pos[1], place_pos[2]) * SE3.Rx(np.pi),
        }
        
        # Compute IK solutions for verification
        q_pick_above, ok1 = self.compute_ik(pick_pos, z_offset=self.approach_height)
        q_pick, ok2 = self.compute_ik(pick_pos, z_offset=0.0)
        q_place_above, ok3 = self.compute_ik(place_pos, z_offset=self.approach_height)
        q_place, ok4 = self.compute_ik(place_pos, z_offset=0.0)
        
        if not (ok1 and ok2 and ok3 and ok4):
            return None
            
        ik_solutions = {
            "home": self.q_home,
            "pick_above": q_pick_above,
            "pick": q_pick,
            "place_above": q_place_above,
            "place": q_place,
        }
        
        result = {
            "poses": poses,
            "ik_solutions": ik_solutions
        }
        
        # Optionally generate joint-space trajectories
        if self.joint_space_controller is not None:
            result["joint_trajectories"] = self.joint_space_controller.plan_pick_and_place_trajectories(
                self.q_home, ik_solutions
            )
        
        return result
        
    def plan_all_trajectories(self, pick_place_pairs):
        """Plan trajectories for all objects."""
        
        trajectories = {}
        
        for name, (pick_pos, place_pos) in pick_place_pairs.items():
            print(f"[INFO] Planning trajectory for {name}...")
            traj = self.plan_pick_and_place(pick_pos, place_pos)
            
            if traj is None:
                print(f"[WARNING] IK failed for {name}, skipping")
                continue
                
            trajectories[name] = traj
            print(f"[SUCCESS] Trajectory planned for {name}")
            
        print(f"\n{'='*60}")
        print(f"Trajectory Planning Complete")
        print(f"Successfully planned: {len(trajectories)}/{len(pick_place_pairs)}")
        print(f"{'='*60}\n")
        
        return trajectories
