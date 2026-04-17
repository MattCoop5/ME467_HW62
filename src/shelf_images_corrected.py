import numpy as np

# --- Image geometry ---
IMG_SIZE = 64
SHELF_THICKNESS = 5  # shelf bar height in pixels
BOX_REGION_TOP = 4  # boxes can stack up to this row
BG_BRIGHTNESS = 0.12  # dark background
SHELF_BRIGHTNESS = 0.75  # light shelf bar


def _draw_shelf(img, shelf_top, rng):
    """Draw the shelf bar with vertical section dividers."""
    shelf_bottom = shelf_top + SHELF_THICKNESS
    img[shelf_top:shelf_bottom, :] = SHELF_BRIGHTNESS
    # Add 2-4 vertical divider lines (shelf section separators).
    # These are straight vertical dark lines within the shelf bar —
    # the CNN must learn that vertical lines are NOT damage (only
    # diagonal lines indicate cracks).
    n_dividers = rng.integers(2, 5)
    for _ in range(n_dividers):
        x = rng.integers(8, IMG_SIZE - 8)
        img[shelf_top:shelf_bottom, x] = rng.uniform(0.25, 0.40)
    return img


def _draw_boxes(img, shelf_top, n_boxes, max_height, rng):
    """Draw darker rectangular boxes sitting on top of the shelf."""
    for _ in range(n_boxes):
        bw = rng.integers(4, 14)
        bh = rng.integers(4, max_height + 1)
        bx = rng.integers(1, IMG_SIZE - bw - 1)
        by_bottom = shelf_top  # box sits on top of shelf
        by_top = max(BOX_REGION_TOP, by_bottom - bh)
        # Boxes are darker than the shelf but lighter than background
        brightness = rng.uniform(0.30, 0.55)
        img[by_top:by_bottom, bx : bx + bw] = brightness
        # Thin darker border for visibility
        border = brightness * 0.65
        img[by_top, bx : bx + bw] = border
        img[by_top:by_bottom, bx] = border
        img[by_top:by_bottom, min(IMG_SIZE - 1, bx + bw - 1)] = border
    return img


def _draw_crack(img, shelf_top, rng):
    """Draw a thin dark crack line contained within the shelf bar."""
    shelf_bottom = shelf_top + SHELF_THICKNESS
    # Crack starts on the left portion of the shelf
    x0 = rng.integers(2, IMG_SIZE // 2)
    y0 = shelf_top + rng.integers(1, SHELF_THICKNESS - 1)
    # Angle: mostly horizontal, slight diagonal
    angle = rng.uniform(-0.3, 0.3)
    length = rng.integers(15, 45)
    for t in range(length):
        x = int(x0 + t * np.cos(angle))
        y = int(y0 + t * np.sin(angle))
        # Clamp crack to the shelf bar region
        if 0 <= x < IMG_SIZE and shelf_top <= y < shelf_bottom:
            img[y, x] = rng.uniform(0.0, 0.15)  # dark crack on light shelf
            # Widen to 2 pixels if still within shelf
            if y + 1 < shelf_bottom:
                img[y + 1, x] = rng.uniform(0.0, 0.20)
    return img


def _box_area_fraction(img, shelf_top):
    """Fraction of the region above the shelf that contains boxes."""
    region = img[BOX_REGION_TOP:shelf_top, :]
    # Boxes have brightness > 0.25 (background is ~0.12)
    return (region > 0.25).mean()


def _random_shelf_top(rng):
    """Random shelf position: base at row 48, jittered ±4."""
    return 48 + rng.integers(-4, 5)


def _add_noise(img, rng):
    """Add brightness variation and Gaussian noise."""
    # Global brightness jitter
    img = img + rng.uniform(-0.04, 0.04)
    # Per-pixel noise
    img = img + rng.normal(0, 0.04, img.shape)
    return np.clip(img, 0, 1)


def generate_normal(rng):
    """Normal shelf: moderate boxes, area fraction < 0.25."""
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 5), max_height=15, rng=rng)
    # Ensure not overloaded
    attempts = 0
    while _box_area_fraction(img, shelf_top) > 0.25 and attempts < 10:
        img[BOX_REGION_TOP:shelf_top, :] = BG_BRIGHTNESS
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=12, rng=rng)
        attempts += 1
    return _add_noise(img, rng)


def generate_damaged(rng):
    """Damaged shelf: similar loading to normal, plus cracks in the shelf bar."""
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 5), max_height=15, rng=rng)
    # Ensure not overloaded
    attempts = 0
    while _box_area_fraction(img, shelf_top) > 0.25 and attempts < 10:
        img[BOX_REGION_TOP:shelf_top, :] = BG_BRIGHTNESS
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=12, rng=rng)
        attempts += 1
    # Draw 1-2 cracks within the shelf bar
    n_cracks = rng.integers(1, 3)
    for _ in range(n_cracks):
        _draw_crack(img, shelf_top, rng)
    return _add_noise(img, rng)


def generate_overloaded(rng):
    """Overloaded shelf: dense tall stacking, area fraction > 0.45."""
    img = np.full((IMG_SIZE, IMG_SIZE), BG_BRIGHTNESS)
    shelf_top = _random_shelf_top(rng)
    _draw_shelf(img, shelf_top, rng)
    _draw_boxes(img, shelf_top, n_boxes=rng.integers(6, 10), max_height=35, rng=rng)
    # Keep adding boxes until overloaded
    attempts = 0
    while _box_area_fraction(img, shelf_top) < 0.45 and attempts < 20:
        _draw_boxes(img, shelf_top, n_boxes=rng.integers(2, 4), max_height=30, rng=rng)
        attempts += 1
    return _add_noise(img, rng)


def generate_dataset(n_per_class=300, seed=42):
    """Generate the full dataset: 300 images per class."""
    rng = np.random.default_rng(seed)
    generators = [generate_normal, generate_damaged, generate_overloaded]
    class_names = ["normal", "damaged", "overloaded"]
    images = []
    labels = []
    for class_idx, gen in enumerate(generators):
        for _ in range(n_per_class):
            img = gen(rng)
            images.append(img)
            labels.append(class_idx)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    # Shuffle
    idx = rng.permutation(len(images))
    images = images[idx]
    labels = labels[idx]
    return images, labels, class_names


images, labels, class_names = generate_dataset(n_per_class=300, seed=42)
print(f"Images: {images.shape} (min={images.min():.2f}, max={images.max():.2f})")
print(f"Labels: {labels.shape}, classes: {class_names}")
print(f"Class distribution: {[int((labels == i).sum()) for i in range(3)]}")
np.savez(
    "shelf_images.npz",
    images=images,
    labels=labels,
    class_names=class_names,
)
print("Saved shelf_images.npz")
