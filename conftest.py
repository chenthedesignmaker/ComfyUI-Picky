"""Root-level pytest configuration.

The repo root carries an `__init__.py` because ComfyUI loads custom nodes as
packages, which makes pytest treat this directory as a package and import it
during test setup. That import is harmless now — the shim uses absolute
imports — but the ComfyUI route module underneath it needs a running ComfyUI,
so it is covered through `video_prompter.api` instead.
"""

collect_ignore = ["__init__.py"]
