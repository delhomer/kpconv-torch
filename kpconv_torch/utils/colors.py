from matplotlib import colors
import numpy as np


def convert_hex_to_rgb(color_codes):
    return np.array(
        [[c * 255 for c in colors.hex2color(color)] for color in color_codes],
        dtype=np.uint8,
    )
