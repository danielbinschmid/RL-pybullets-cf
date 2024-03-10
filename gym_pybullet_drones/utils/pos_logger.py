import numpy as np
import os 
import shutil
from typing import Optional

class PosLoggerConfig:

    def __init__(self, max_len=1000, log_folder: str="./positions") -> None:
        self.max_len=max_len
        self.log_folder = log_folder


def load_positions(log_folder: str) -> np.ndarray:
    all_positions = []
    all_velocities = []
    filenames = os.listdir(log_folder)

    filenames_pos = [fname for fname in filenames if fname.split('.')[1] == 'pos']
    filenames_vel = [fname for fname in filenames if fname.split('.')[1] == 'vel']

    filenames_pos = sorted(filenames_pos, key=lambda x: int(x.split('.')[0]))
    filenames_vel = sorted(filenames_vel, key=lambda x: int(x.split('.')[0]))

    for fname in filenames_pos:
        fname_full = os.path.join(log_folder, fname)
        positions = np.load(fname_full)
        all_positions.append(positions)
    all_positions = np.concatenate(all_positions)

    for fname in filenames_vel:
        fname_full = os.path.join(log_folder, fname)
        velocities = np.load(fname_full)
        all_velocities.append(velocities)
    all_velocities = np.concatenate(all_velocities)

    return all_positions, all_velocities

class PositionLogger:
    def __init__(self, config: PosLoggerConfig) -> None:
        self.max_len = config.max_len
        self.n_logged_files = 0
        self.log_folder = config.log_folder
        self.positions = []
        self.velocities = []

        if os.path.isfile(self.log_folder):
            os.remove(self.log_folder)
        if os.path.isdir(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.makedirs(self.log_folder, exist_ok=True)

    def log_position(self, pos, vel):
        pos = np.array(pos)
        self.positions.append(pos.reshape((1, 3)))
        
        vel = np.array(vel)
        self.velocities.append(vel.reshape((1,3)))

        assert(len(self.positions) == len(self.velocities))

        if len(self.positions) >= self.max_len:
            self._log()
    
    def _log(self):
        pos_np = np.vstack(self.positions)
        vel_np = np.vstack(self.velocities)
        fname_pos = os.path.join(
            self.log_folder,
            f'{self.n_logged_files}.pos.npy'
        )
        fname_vel = os.path.join(
            self.log_folder,
            f'{self.n_logged_files}.vel.npy'
        )
        np.save(
            fname_pos, pos_np
        )
        np.save(
            fname_vel,vel_np
        )
        self.positions = []
        self.velocities = []
        self.n_logged_files += 1

    def flush(self):
        if len(self.positions) > 0:
            self._log()
    
    def load_all(self) -> np.ndarray: 
        return load_positions(self.log_folder)[0]
    
