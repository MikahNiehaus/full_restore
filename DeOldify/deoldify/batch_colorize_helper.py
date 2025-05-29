
from .visualize import get_image_colorizer
# Add your image colorization logic here, e.g.:
if __name__ == '__main__':
    colorizer = get_image_colorizer(artistic=False)
    # ...process images in ../inputs and save to ../outputs...
