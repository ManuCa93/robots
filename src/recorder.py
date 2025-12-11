"""Data recording for trajectory analysis"""

import time
import numpy as np

class DataRecorder:
    """Records robot end-effector positions and timestamps."""
    
    def __init__(self):
        # Struttura: { 'nome_cubo': { 'positions': [[x,y,z], ...], 'times': [t0, t1...] } }
        self.history = {}
        self.start_time = time.time()

    def log_pose(self, name, robot_pose):
        """
        Registra la posizione corrente dell'End-Effector.
        
        Args:
            name: Nome del cubo che si sta manipolando (es. 'red1')
            robot_pose: Posa SE3 o matrice 4x4 o vettore posizione
        """
        if name is None: 
            return
        
        # Estrai la traslazione (x, y, z) dalla posa
        # Se robot_pose è un oggetto SE3 di spatialmath, usa .t
        if hasattr(robot_pose, 't'):
            pos = robot_pose.t
        # Se è una matrice numpy 4x4
        elif isinstance(robot_pose, np.ndarray) and robot_pose.shape == (4, 4):
            pos = robot_pose[:3, 3]
        # Se è già un vettore posizione
        else:
            pos = np.array(robot_pose)
            
        # Calcola il tempo relativo dall'inizio
        current_t = time.time() - self.start_time
        
        # Inizializza la lista per questo cubo se non esiste
        if name not in self.history:
            self.history[name] = {
                'positions': [],
                'times': []
            }
            
        # Salva i dati
        self.history[name]['positions'].append(pos)
        self.history[name]['times'].append(current_t)

    def get_data(self):
        """Restituisce tutto lo storico registrato."""
        return self.history