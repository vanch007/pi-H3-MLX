"""FlowEulerODEScheduler (LakonLab) — the few-step pi-H3 sampler, in the sigma convention (sigma=1 is noise)."""

import numpy as np


class FlowEulerSchedule:
    def __init__(self, num_steps: int, shift: float, max_raw_t: float, final_step_size_scale: float = 0.6,
                 min_raw_t: float = 0.0):
        raw = np.linspace(
            max_raw_t,
            (max_raw_t - min_raw_t) * final_step_size_scale / (num_steps - 1 + final_step_size_scale) + min_raw_t,
            num_steps, dtype=np.float32)
        sig = (shift * raw / (1 + (shift - 1) * raw)).astype(np.float32)
        self.sigmas = np.concatenate([sig, np.zeros(1, np.float32)])

    def __len__(self):
        return len(self.sigmas) - 1

    def step(self, x, velocity, i: int):
        """Euler step x_{i+1} = x_i + u * (sigma_{i+1} - sigma_i); computed in float32."""
        return x + velocity * float(self.sigmas[i + 1] - self.sigmas[i])


def pih3_schedules(num_steps: int):
    """The video (shift 12) and audio (shift 3) schedules of the pi-H3 2-/4-step adapters."""
    max_raw_t = {2: 0.93, 4: 0.96}[num_steps]
    return FlowEulerSchedule(num_steps, 12.0, max_raw_t), FlowEulerSchedule(num_steps, 3.0, max_raw_t)
