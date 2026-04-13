"""Generate synthetic 64x64 grayscale shelf images for CNN training.

Creates 900 images total (300 per class):
- 0: normal
- 1: damaged
- 2: overloaded

Saves an NPZ file named ``shelf_images.npz`` containing:
- images: uint8 array, shape (900, 64, 64)
- labels: int64 array, shape (900,)
- class_names: string array, shape (3,)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


IMG_SIZE = 64
CLASSES = ("normal", "damaged", "overloaded")
SAMPLES_PER_CLASS = 300


def add_noise(
    img: np.ndarray, rng: np.random.Generator, sigma: float = 6.0
) -> np.ndarray:
    """Add Gaussian sensor noise and clip to [0, 255]."""
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def draw_line(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    value: int,
    thickness: int = 1,
) -> None:
    """Draw a line using linear interpolation."""
    steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.linspace(x0, x1, steps).astype(int)
    ys = np.linspace(y0, y1, steps).astype(int)

    for x, y in zip(xs, ys, strict=False):
        y0_t = max(0, y - thickness // 2)
        y1_t = min(IMG_SIZE, y + thickness // 2 + 1)
        x0_t = max(0, x - thickness // 2)
        x1_t = min(IMG_SIZE, x + thickness // 2 + 1)
        img[y0_t:y1_t, x0_t:x1_t] = value


def draw_rect(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value: int,
    fill: bool = True,
    border: int = 1,
) -> None:
    """Draw a rectangle (filled or outlined)."""
    x0 = np.clip(x, 0, IMG_SIZE)
    y0 = np.clip(y, 0, IMG_SIZE)
    x1 = np.clip(x + w, 0, IMG_SIZE)
    y1 = np.clip(y + h, 0, IMG_SIZE)

    if x0 >= x1 or y0 >= y1:
        return

    if fill:
        img[y0:y1, x0:x1] = value
    else:
        img[y0 : min(y0 + border, y1), x0:x1] = value
        img[max(y1 - border, y0) : y1, x0:x1] = value
        img[y0:y1, x0 : min(x0 + border, x1)] = value
        img[y0:y1, max(x1 - border, x0) : x1] = value


def base_shelf_scene(
    rng: np.random.Generator,
    shelf_y: int,
) -> np.ndarray:
    """Create a simple front-view shelf scene with one horizontal shelf bar."""
    bg = int(rng.integers(18, 38))
    img = np.full((IMG_SIZE, IMG_SIZE), bg, dtype=np.uint8)

    # Mild illumination gradient.
    yy, xx = np.indices((IMG_SIZE, IMG_SIZE))
    grad = ((xx / (IMG_SIZE - 1)) * rng.uniform(2.0, 7.0)) + (
        (yy / (IMG_SIZE - 1)) * rng.uniform(1.0, 5.0)
    )
    img = np.clip(img.astype(np.float32) + grad, 0, 255).astype(np.uint8)

    # Light shelf bar spanning full width.
    shelf_thickness = int(rng.integers(4, 7))
    shelf_val = int(rng.integers(165, 215))
    draw_rect(img, 0, shelf_y, IMG_SIZE, shelf_thickness, shelf_val, fill=True)

    return img


def stack_boxes(
    img: np.ndarray,
    rng: np.random.Generator,
    shelf_y: int,
    box_count_range: tuple[int, int],
    box_w_range: tuple[int, int],
    box_h_range: tuple[int, int],
    tight_spacing: bool,
) -> None:
    """Draw darker boxes stacked above the shelf bar."""
    n_boxes = int(rng.integers(box_count_range[0], box_count_range[1]))

    # Track current stack heights per x-coordinate to allow stacking.
    top_profile = np.full(IMG_SIZE, shelf_y - 1, dtype=int)

    for _ in range(n_boxes):
        w = int(rng.integers(box_w_range[0], box_w_range[1]))
        h = int(rng.integers(box_h_range[0], box_h_range[1]))
        x0 = int(rng.integers(0, IMG_SIZE - w + 1))
        x1 = x0 + w

        # Place this box on top of the highest occupied point in its footprint.
        base_y = int(top_profile[x0:x1].min())
        y1 = base_y
        y0 = y1 - h

        if y0 < 1:
            continue

        val = int(rng.integers(55, 140))  # darker than shelf bar
        draw_rect(img, x0, y0, w, h, val, fill=True)

        if rng.random() < 0.5:
            edge_val = int(np.clip(val + rng.integers(10, 30), 0, 255))
            draw_rect(img, x0, y0, w, h, edge_val, fill=False, border=1)

        # Update occupancy profile.
        top_profile[x0:x1] = np.minimum(
            top_profile[x0:x1], y0 - int(rng.integers(0, 2))
        )

        # Optional local jitter to make overloaded shelves denser and less regular.
        if tight_spacing and rng.random() < 0.5:
            x2 = max(0, min(IMG_SIZE - w, x0 + int(rng.integers(-2, 3))))
            y2 = max(1, y0 - int(rng.integers(0, 3)))
            draw_rect(
                img,
                x2,
                y2,
                w,
                max(2, h - int(rng.integers(0, 3))),
                int(rng.integers(50, 130)),
                fill=True,
            )
            top_profile[x2 : x2 + w] = np.minimum(top_profile[x2 : x2 + w], y2 - 1)


def add_normal_boxes(img: np.ndarray, rng: np.random.Generator, shelf_y: int) -> None:
    """Normal class: modest loading above shelf."""
    stack_boxes(
        img,
        rng,
        shelf_y=shelf_y,
        box_count_range=(7, 12),
        box_w_range=(7, 13),
        box_h_range=(6, 12),
        tight_spacing=False,
    )


def add_damage(img: np.ndarray, rng: np.random.Generator, shelf_y: int) -> None:
    """Damaged class: modest loading with a thin dark crack through the shelf bar."""
    stack_boxes(
        img,
        rng,
        shelf_y=shelf_y,
        box_count_range=(6, 11),
        box_w_range=(7, 13),
        box_h_range=(6, 12),
        tight_spacing=False,
    )

    # Thin dark crack through the shelf bar with randomized position and slight slope.
    crack_y = shelf_y + int(rng.integers(1, 4))
    x0 = int(rng.integers(0, 14))
    x1 = int(rng.integers(50, 64))
    y1 = crack_y + int(rng.integers(-1, 2))
    draw_line(img, x0, crack_y, x1, y1, int(rng.integers(10, 40)), thickness=1)


def add_overload(img: np.ndarray, rng: np.random.Generator, shelf_y: int) -> None:
    """Overloaded class: dense, tall stacking above shelf."""
    stack_boxes(
        img,
        rng,
        shelf_y=shelf_y,
        box_count_range=(18, 30),
        box_w_range=(5, 10),
        box_h_range=(9, 17),
        tight_spacing=True,
    )


def generate_sample(class_name: str, rng: np.random.Generator) -> np.ndarray:
    """Generate one sample for a specific class."""
    base_y = 44
    shelf_y = int(base_y + rng.integers(-4, 5))
    img = base_shelf_scene(rng, shelf_y=shelf_y)

    if class_name == "normal":
        add_normal_boxes(img, rng, shelf_y=shelf_y)
    elif class_name == "damaged":
        add_damage(img, rng, shelf_y=shelf_y)
    elif class_name == "overloaded":
        add_overload(img, rng, shelf_y=shelf_y)
    else:
        raise ValueError(f"Unknown class: {class_name}")

    return add_noise(img, rng, sigma=float(rng.uniform(4.0, 8.0)))


def build_dataset(
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = 467,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build full dataset and return (images, labels, class_names)."""
    rng = np.random.default_rng(seed)

    image_list: list[np.ndarray] = []
    label_list: list[int] = []

    for class_idx, class_name in enumerate(CLASSES):
        for _ in range(samples_per_class):
            image_list.append(generate_sample(class_name, rng))
            label_list.append(class_idx)

    images = np.stack(image_list).astype(np.uint8)
    labels = np.array(label_list, dtype=np.int64)
    class_names = np.array(CLASSES)

    # Shuffle dataset so classes are mixed.
    perm = rng.permutation(len(labels))
    images = images[perm]
    labels = labels[perm]

    return images, labels, class_names


def save_dataset(output_path: Path) -> Path:
    """Generate and save the shelf dataset to an .npz file."""
    images, labels, class_names = build_dataset()
    np.savez_compressed(
        output_path,
        images=images,
        labels=labels,
        class_names=class_names,
    )
    return output_path


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parent / "shelf_images.npz"
    saved = save_dataset(out_file)
    print(f"Saved dataset to: {saved}")
    print("images shape:", np.load(saved)["images"].shape)
