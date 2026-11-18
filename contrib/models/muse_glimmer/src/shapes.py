# SPDX-License-Identifier: Apache-2.0
"""Fixed vision shapes for Muse-Glimmer on Neuron (one compiled vision graph each).

The processor resizes every input to one of these shapes; the vision graph is
picked by patch count, so every shape has its own patch count.

Images go to the shape whose aspect ratio is closest to the input, so a photo
keeps its proportions instead of being squashed into a square. The 4:3 shapes
are the grid that the transformers processor itself picks for a 640x480 photo.

  image 448x448 (square)    -> grid [1, 32, 32] -> 1024 patches -> 256 embeddings
  image 644x476 (landscape) -> grid [1, 34, 46] -> 1564 patches -> 391 embeddings
  image 476x672 (portrait)  -> grid [1, 48, 34] -> 1632 patches -> 408 embeddings
  video: VIDEO_FRAMES frames at VIDEO_W x VIDEO_H -> grid [4, 20, 26] -> 2080 patches -> 520

The portrait shape is 476x672, not 476x644, so that it does not share the
landscape shape's patch count.
"""

import math

MERGE_FACTOR = 4  # 2x2 patches per embedding (vision_config.merge_size == 2)
PATCH = 14
IMAGE_SHAPES = ((448, 448), (644, 476), (476, 672))  # (width, height) in pixels
VIDEO_FRAMES, VIDEO_W, VIDEO_H = 8, 480, 368
VIDEO_GRID = (4, 20, 26)  # transformers processor grid for 8 frames at 480x368


def _grid(width: int, height: int) -> tuple[int, int, int]:
    return (1, height // PATCH, width // PATCH)  # (T, H, W) in patches


def _count(grid: tuple[int, int, int]) -> int:
    return grid[0] * grid[1] * grid[2]


IMAGE_GRIDS = tuple(_grid(w, h) for w, h in IMAGE_SHAPES)
GRIDS = {_count(g): g for g in IMAGE_GRIDS + (VIDEO_GRID,)}  # patches -> grid
assert len(GRIDS) == len(IMAGE_SHAPES) + 1, (
    "every vision shape needs its own patch count"
)
IMAGE_TOKENS = (
    max(_count(g) for g in IMAGE_GRIDS) // MERGE_FACTOR
)  # most tokens of one image
IMAGE_TOKENS_MIN = (
    min(_count(g) for g in IMAGE_GRIDS) // MERGE_FACTOR
)  # fewest tokens of one item (a video has more)
VIDEO_TOKENS = _count(VIDEO_GRID) // MERGE_FACTOR  # 520
IMAGE_SIZE_MOST_TOKENS = max(
    IMAGE_SHAPES, key=lambda s: _count(_grid(*s))
)  # (width, height)


def pick_image_shape(width: int, height: int) -> tuple[int, int]:
    """Return the fixed (width, height) whose aspect ratio is closest to the input's."""
    r = math.log(width / height)
    return min(IMAGE_SHAPES, key=lambda s: abs(math.log(s[0] / s[1]) - r))
