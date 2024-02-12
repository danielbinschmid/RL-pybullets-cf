import numpy as np
import os 
import shutil
class PosLoggerConfig:

    def __init__(self, max_len=1000, log_folder: str="./positions") -> None:
        self.max_len=max_len
        self.log_folder = log_folder


def load_positions(log_folder: str) -> np.ndarray:
    all_positions = []
    filenames = os.listdir(log_folder)
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0]))
    for fname in filenames:
        fname_full = os.path.join(log_folder, fname)
        positions = np.load(fname_full)
        all_positions.append(positions)
    all_positions = np.concatenate(all_positions)
    return all_positions

class PositionLogger:
    def __init__(self, config: PosLoggerConfig) -> None:
        self.max_len = config.max_len
        self.n_logged_files = 0
        self.log_folder = config.log_folder
        self.positions = []

        if os.path.isfile(self.log_folder):
            os.remove(self.log_folder)
        if os.path.isdir(self.log_folder):
            shutil.rmtree(self.log_folder)
        os.makedirs(self.log_folder, exist_ok=True)

    def log_position(self, pos):
        pos = np.array(pos)
        self.positions.append(pos.reshape((1, 3)))
        if len(self.positions) >= self.max_len:
            self._log()
    
    def _log(self):
        pos_np = np.vstack(self.positions)
        fname = os.path.join(
            self.log_folder,
            f'{self.n_logged_files}.npy'
        )
        np.save(
            fname, pos_np
        )
        self.positions = []
        self.n_logged_files += 1

    def flush(self):
        if len(self.positions) > 0:
            self._log()
    
    def load_all(self) -> np.ndarray: 
        return load_positions(self.log_folder)
    
