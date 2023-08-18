# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
sys.path.append("../python/visual_search/")
from visual_search import VisualSearch
from models import *
import json
from glob import glob
import ipyplot
from matplotlib import pyplot as plt
from tqdm import tqdm

# Creating a collection to keep the images 
# -------------
#
# In this case we are using MOBILE_NET_V2 as the feature extractor. 

api = VisualSearch(bearer_token="secrettoken", address="http://localhost:8890")
upsert_collection = UpsertCollection(
    config=GenericModelConfig(
        model_architecture=ModelArchitecture.MOBILE_NET_V2
    ), 
    name="images"
)
response = api.upsert_collection(upsert_collection)

# Indexing local images
# -----------
#
# It is possible to index local images using `ImageBytes` or `ImageSource(url="link_to_image")`

for img_path in tqdm(sorted(glob("../../images/ecommerce-images/data/Apparel/Boys/Images/images_with_product_ids/*.jpg"))):
    image_id = img_path.split("/")[-1].split(".")[0]
    with open(img_path, "rb") as inp:
        image_bytes = list(inp.read())
    image_source = ImageSource(image_bytes=ImageBytes(image_bytes))
    add_image = AddImage(collection_name="images", id=image_id, source=image_source)
    resp = api.add_image(add_image)

# Searching for a cat 
# -----------

img = plt.imread("../../images/ecommerce-images/data/Apparel/Boys/Images/images_with_product_ids/10054.jpg")
ipyplot.plot_images([img], img_width=300)

with open("../../images/ecommerce-images/data/Apparel/Boys/Images/images_with_product_ids/10054.jpg", "rb") as inp:
    image_bytes = list(inp.read())
image_source = ImageSource(image_bytes=ImageBytes(image_bytes))
search_image = SearchImage(collection_name="images", n_results=8, source=image_source)
# %time search_results = json.loads(api.search_image(search_image).content)
search_results

images_paths = []
for result in search_results["results"]:
    fn = "../../images/ecommerce-images/data/Apparel/Boys/Images/images_with_product_ids/{}.jpg".format(result["id"])
    img = plt.imread(fn)
    images_paths.append(img)    
ipyplot.plot_images(images_paths, img_width=200)


