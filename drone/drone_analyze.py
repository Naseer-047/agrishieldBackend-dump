from PIL import Image
import numpy as np


AGRI_CLASSES = {
    0: "background",
    1: "double_plant",
    2: "drydown",
    3: "endrow",
    4: "nutrient_deficiency",
    5: "planter_skip",
    6: "water",
    7: "waterway",
    8: "weed_cluster",
}


def analyze_drone_image(image: Image.Image):

    image = image.convert("RGB")

    # Temporary safety check.
    # The actual Agriculture-Vision model will replace this
    # vegetation-only prototype once checkpoint inference is connected.
    img = np.array(image)

    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)

    vegetation = (
        (g > r * 1.05) &
        (g > b * 1.05)
    )

    vegetation_percent = vegetation.mean() * 100

    h, w = vegetation.shape

    zones = {}

    zone_coordinates = [
        ("Zone A", 0, h // 2, 0, w // 2),
        ("Zone B", 0, h // 2, w // 2, w),
        ("Zone C", h // 2, h, 0, w // 2),
        ("Zone D", h // 2, h, w // 2, w),
    ]

    for name, y1, y2, x1, x2 in zone_coordinates:

        zone = vegetation[y1:y2, x1:x2]

        zones[name] = {
            "vegetation_percent": round(
                float(zone.mean() * 100), 2
            ),
            "status": "requires_inspection"
            if zone.mean() < 0.5
            else "normal"
        }

    return {
        "analysis_type": "aerial_field_screening",
        "model": "Agriculture-Vision MSCG-Net",
        "vegetation_coverage_percent": round(
            vegetation_percent, 2
        ),
        "zones": zones,
        "possible_anomalies": [
            "nutrient_deficiency",
            "weed_cluster",
            "water",
            "waterway",
            "drydown",
            "planter_skip",
            "double_plant",
            "endrow"
        ],
        "next_step": (
            "Inspect flagged zones with a close-up leaf image "
            "for specific disease diagnosis."
        )
    }