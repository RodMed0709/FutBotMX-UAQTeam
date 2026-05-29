import argparse
import logging
import json
import os
import sys
import random

from pathlib import Path

from src.core.io import FrameGenerator
from src.core.detector import ZeroShotDetector
from src.core.utils import get_abs_path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrator Config-Driven - Copa FutBotMX"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/configs.json",
        help="Path to the experiment's JSON configuration file",
    )

    return parser.parse_args()


def get_random_video(dataset_root: str) -> str:
    """Searches for video files recursively and returns one at random."""
    root_path = get_abs_path(dataset_root)
    if not root_path.exists():
        raise FileNotFoundError(f"The data directory does not exist: {dataset_root}")

    # Buscar .MOV y .mp4 (case-insensitive)
    videos = []
    for ext in ("*.mov", "*.MOV", "*.mp4", "*.MP4"):
        videos.extend(root_path.rglob(ext))

    if not videos:
        raise FileNotFoundError(f"No videos were found in {dataset_root}")

    chosen_video = random.choice(videos)
    logger.info(
        f"Randomly selected video: {chosen_video.name} (from {len(videos)} available)"
    )
    return str(chosen_video)


def main():
    # 1. Load configuration from  args
    # dataset_root = os.getenv("DATASET_ROOT", "data/raw")
    args = parse_args()
    logger.info(f"Initialising experiment with configuration: {args.config}")

    CONFIG_PATH = get_abs_path(args.config)

    # 2. Load configuration json files
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Configuration file not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

        try:
            # 3. Select sample video and FrameGenerator configuration
            dataset_root = config.get("model", {}).get("data_path", "data/raw")
            frames_to_process = config.get("io", {}).get("sample_frames", None)
            video_path = get_random_video(dataset_root)

            generator = FrameGenerator(
                video_path=video_path, sample_frames=frames_to_process
            )

            sam3_path = config.get("model", {}).get("sam3_path", "assets/sam3")
            prompts = config.get("model", {}).get(
                "prompts", ["robot", "orange ball", "green floor"]
            )
            logger.info("Initializing SAM 3 ZeroShotDetector...")
            detector = ZeroShotDetector(model_path=sam3_path)

            detector.load_model()

        except Exception as e:
            logger.error(f"Error in I/O Initialization: {e}")
            sys.exit(1)

        # 4. Testing main Loop
        logger.info(f"Frame retrieving started")

        for frame_idx, frame_rgb in generator:
            height, width, channels = frame_rgb.shape
            logger.info(
                f"Retrieved frame: {frame_idx:04d} | Resolution:{width}x{height} | Channels: {channels}"
            )

            try:
                detections = detector.predict_zero_shot(frame_rgb, prompts)

                logger.info(
                    f"   -> Found {len(detections)} object masks in frame {frame_idx:04d}"
                )

            except Exception as e:
                logger.error(f"Inference failed at frame {frame_idx:04d}: {e}")

        logger.info("Config-Driven FrameGenerator Test successfully finished")


if __name__ == "__main__":
    main()
