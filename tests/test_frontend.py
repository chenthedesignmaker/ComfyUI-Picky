"""Run client state/transport regression tests as part of the normal suite."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("filename", ["frontend_save.test.cjs", "frontend_shot_ideas.test.cjs", "frontend_camera_path.test.cjs", "frontend_usage.test.cjs", "frontend_compare.test.cjs", "frontend_take_context.test.cjs", "frontend_video_loras.test.cjs", "frontend_prompt_sections.test.cjs", "frontend_pipeline_progress.test.cjs"])
def test_frontend_save_and_refresh_regressions(filename):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for client regression tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name(filename))],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
