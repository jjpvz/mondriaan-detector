from processing_tools import processing_image, img_import_resize
from settings import folder_path_single
import pandas as pd

img_set, paths = img_import_resize(folder_path_single)

features_list = processing_image(img_set, paths)

dataset = pd.DataFrame(features_list)

print(dataset)