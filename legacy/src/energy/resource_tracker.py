import os
import time
import threading
from typing import Optional

import psutil


class ResourceTracker:
    """
    Tracks wall-clock time, peak RAM, optional GPU memory, and optionally
    energy/carbon via CodeCarbon.

    If CodeCarbon fails or is unavailable, energy/carbon are set to 0.0.
    """

    def __init__(
        self,
        track_energy: bool = True,
        sample_interval_sec: float = 0.05,
        project_name: str = "sustainable-qrc-benchmark",
    ) -> None:
        self.track_energy = track_energy
        self.sample_interval_sec = sample_interval_sec
        self.project_name = project_name

        self.elapsed_time_sec: float = 0.0
        self.peak_ram_gb: float = 0.0
        self.peak_gpu_mem_gb: float = 0.0
        self.energy_kwh: float = 0.0
        self.carbon_kgco2e: float = 0.0

        self._process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._tracker = None

    def _sample_ram_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                rss = self._process.memory_info().rss

                for child in self._process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass

                self.peak_ram_gb = max(self.peak_ram_gb, rss / (1024 ** 3))
            except psutil.Error:
                pass

            time.sleep(self.sample_interval_sec)

    def __enter__(self) -> "ResourceTracker":
        self._start_time = time.perf_counter()
        self._stop_event.clear()

        self._thread = threading.Thread(target=self._sample_ram_loop, daemon=True)
        self._thread.start()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        if self.track_energy:
            try:
                from codecarbon import EmissionsTracker

                self._tracker = EmissionsTracker(
                    project_name=self.project_name,
                    save_to_file=False,
                    log_level="error",
                    measure_power_secs=1,
                )
                self._tracker.start()
            except Exception:
                self._tracker = None

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._start_time is not None:
            self.elapsed_time_sec = time.perf_counter() - self._start_time

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        try:
            import torch

            if torch.cuda.is_available():
                self.peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        except Exception:
            self.peak_gpu_mem_gb = 0.0

        if self._tracker is not None:
            try:
                emissions = self._tracker.stop()
                self.carbon_kgco2e = float(emissions) if emissions is not None else 0.0

                data = getattr(self._tracker, "final_emissions_data", None)
                energy = getattr(data, "energy_consumed", None)
                self.energy_kwh = float(energy) if energy is not None else 0.0
            except Exception:
                self.energy_kwh = 0.0
                self.carbon_kgco2e = 0.0