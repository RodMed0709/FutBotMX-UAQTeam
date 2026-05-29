import argparse
import logging
import json
import os
import sys
import random

from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))

from core.io import FrameGenerator


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
    root_path = Path(dataset_root)
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
    # 1. Load env and config args
    load_dotenv()
    dataset_root = os.getenv("DATASET_ROOT", "data/raw")

    args = parse_args()
    logger.info(f"Initialising experiment with configuration: {args.config}")

    # 2. Load configuration json files
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    with open(args.config, "r") as f:
        config = json.load(f)

        try:
            # 3. Select sample video and FrameGenerator configuration
            video_path = get_random_video(dataset_root)
            frames_to_process = config.get("io", {}).get("sample_frames", None)

            generator = FrameGenerator(
                video_path=video_path, sample_frames=frames_to_process
            )
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

        logger.info("Config-Driven FrameGenerator Test successfully finished")


if __name__ == "__main__":
    main()
