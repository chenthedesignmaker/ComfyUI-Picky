"""ComfyUI custom node entry point.

ComfyUI loads this file by path (`spec_from_file_location`) and does not put
the custom-node directory on `sys.path`, so the package underneath would not be
importable by name. Adding this directory ourselves means everything below can
use plain absolute imports and stays importable outside ComfyUI too — by
pytest, by the CLI, by anything.

Only `video_prompter` is exposed by that, which is distinctive enough not to
shadow anything of ComfyUI's. Both entry points live inside the package rather
than beside this file for the same reason: a top-level `server.py` here would
be a loaded gun.
"""

import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from video_prompter.comfy_node import VideoPromptNode  # noqa: E402
from video_prompter.camera_frames import CameraFrames
from video_prompter.small_faces import SmallFaceRefine
from video_prompter.seedvr2 import SeedVR2FaceBlend

logger = logging.getLogger("ComfyUI.VideoPrompter")

NODE_CLASS_MAPPINGS = {"VideoPromptNode": VideoPromptNode, "VideoPrompterCameraFrames": CameraFrames,
                       "VideoPrompterSmallFaceRefine": SmallFaceRefine,
                       "VideoPrompterSeedVR2FaceBlend": SeedVR2FaceBlend}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoPromptNode": "Picky Prompt (MiniMax H3)",
                              "VideoPrompterCameraFrames": "Picky Camera Frames",
                              "VideoPrompterSmallFaceRefine": "Picky Small Face Refine",
                              "VideoPrompterSeedVR2FaceBlend": "Picky SeedVR2 Face Blend"}

WEB_DIRECTORY = "./js"

try:
    from video_prompter import comfy_routes  # noqa: F401  (registers routes)
except Exception as error:  # pragma: no cover - depends on the host
    logger.warning(f"Video prompter routes not registered: {error}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
