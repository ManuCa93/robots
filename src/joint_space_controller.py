"""Joint-Space Trajectory Motion Control"""

import numpy as np
from scipy.interpolate import CubicSpline


class JointSpaceController:
    """Implements joint-space trajectory control using cubic splines."""
    
    def __init__(self, duration=2.0, n_points=100):
        """
        Initialize joint-space controller.
        
        Args:
            duration: Duration of trajectory in seconds
            n_points: Number of points in the trajectory
        """
        self.duration = duration
        self.n_points = n_points
        
    def generate_trajectory(self, q_start, q_goal):
        """
        Generate a smooth joint-space trajectory from q_start to q_goal.
        
        Args:
            q_start: Starting joint configuration (7 joints for Panda)
            q_goal: Goal joint configuration (7 joints for Panda)
        """
        t = np.linspace(0, self.duration, self.n_points)
        n_joints = len(q_start)
        traj = np.zeros((self.n_points, n_joints))
        
        for i in range(n_joints):
            # Cubic spline between start and goal for joint i
            cs = CubicSpline([0, self.duration], [q_start[i], q_goal[i]])
            traj[:, i] = cs(t)
        
        return traj
    
    def plan_pick_and_place_trajectories(self, q_home, ik_solutions):
        """
        Plan all joint-space trajectories for a pick-and-place operation.
        
        Args:
            q_home: Home configuration
            ik_solutions: Dictionary with IK solutions for waypoints
        """
        trajectories = {
            "home_to_pick_above": self.generate_trajectory(q_home, ik_solutions["pick_above"]),
            "pick_above_to_pick": self.generate_trajectory(ik_solutions["pick_above"], ik_solutions["pick"]),
            "pick_to_pick_above": self.generate_trajectory(ik_solutions["pick"], ik_solutions["pick_above"]),
            "pick_above_to_place_abv": self.generate_trajectory(ik_solutions["pick_above"], ik_solutions["place_above"]),
            "place_abv_to_place": self.generate_trajectory(ik_solutions["place_above"], ik_solutions["place"]),
            "place_to_place_abv": self.generate_trajectory(ik_solutions["place"], ik_solutions["place_above"]),
            "place_abv_to_home": self.generate_trajectory(ik_solutions["place_above"], q_home),
        }
        
        return trajectories
