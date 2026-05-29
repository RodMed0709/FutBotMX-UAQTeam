import logging
import time
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

logger = logging.getLogger(__name__)


class ZeroShotDetector:
    """It encapsulates the logic of SAM 3 for zero-shot inference.
    It offloads tensor processing and memory management to the GPU.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.procesor = None
        self.model = None

    def load_model(self):
        """Load the model and the processor into the GPU memory."""

        if self.model is not None:
            logger.warning("The model is already loaded into the memory")
            return

        logger.info(
            f"Starting loading of SAM 3 from {self.model_path} into {self.device}..."
        )
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(self.model_path)

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()

        tiempo_carga = time.time() - t0
        parametros = sum(p.numel() for p in self.model.parameters()) / 1e6

        logger.info(f"Loading completed in  {tiempo_carga:.1f}s.")
        logger.info(f"Parameters: {parametros:.1f}M | Device: {self.model.device}")

    @torch.no_grad()
    def predict_zero_shot(self, frame_rgb: np.ndarray, prompts: list) -> list:
        """
        Performs zero-shot inference on a frame using text prompts.
         Returns a list of dictionaries containing Boolean masks and confidence scores.
        """
        if self.model is None:
            raise RuntimeError(
                "You must call `load_model()` before making a prediction."
            )

        image_pil = Image.fromarray(frame_rgb)

        session = self.processor.init_video_session(
            video=[image_pil],
            inference_device=self.device,
            dtype=torch.bfloat16,
        )

        for text in prompts:
            session = self.processor.add_text_prompt(session, text=text)

        out = self.model(inference_session=session, frame_idx=0)

        detections = []
        for obj_id in out.object_ids:
            mask_tensor = out.obj_id_to_mask[obj_id]
            score = float(out.obj_id_to_score.get(obj_id, 0.0))

            mask_np = mask_tensor.squeeze().cpu().float().numpy() > 0.0

            detections.append(
                {
                    "id_interno": int(obj_id),
                    "mask": mask_np,
                    "score": score,
                }
            )

        return detections
