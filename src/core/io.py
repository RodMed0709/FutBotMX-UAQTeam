from logging import Logger
import decord
import logging
import numpy as np

logger: Logger = logging.getLogger(__name__)


class FrameGenerator:
    """Manage efficient video playback using decord. It allows you to sample a specific number of frames for quick experimentation, or process 100% of the video if no limit is specified."""

    def __init__(self, video_path: str, sample_frames: int = None):
        self.video_path = video_path
        self.sample_frames = sample_frames
        try:
            self.vr = decord.VideoReader(video_path)
        except Exception as e:
            raise ValueError(
                f"Critical Error: decord can't open the video {video_path}"
            )

        self.total_frames = len(self.vr)
        self.fps = self.vr.get_avg_fps()

        if self.sample_frames and 0 < self.sample_frames < self.total_frames:
            self.frame_indices = np.linspace(
                0, self.total_frames - 1, self.sample_frames, dtype=int
            ).tolist()
            logger.info(
                f"Video: {video_path} | EXPERIMENTAL MODE: retrieving {self.sample_frames} frames."
            )
        else:
            self.frame_indices = list(range(self.total_frames))
            logger.info(
                f"Video: {video_path} | PRODUCTION MODE: Processing {self.total_frames} frames @ {self.fps:.1f} FPS."
            )

    def __iter__(self):
        """Frame generator iterative loop; Consider the calculated indices"""
        for idx in self.frame_indices:
            frame_np = self.vr[idx].asnumpy()
            yield idx, frame_np

    def get_duration(self):
        return self.total_frames / self.fps if self.fps > 0 else 0.0
