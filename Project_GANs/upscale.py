import glob
import shutil

from PIL import Image
from super_image import RcanModel, ImageLoader

model = RcanModel.from_pretrained('eugenesiow/rcan-bam', scale=4)

for index, file in enumerate(glob.glob('.\\images\\*.jpg')):
    image1 = Image.open(file)
    input = ImageLoader.load_image(image1)
    predictions = model(input)
    ImageLoader.save_image(predictions, f'.\\scaled_4x\\{index}.jpg')

shutil.make_archive("images", "zip", ".\\images")
