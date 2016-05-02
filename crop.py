from glob import glob
from PIL import Image
i=1
for file_count, file_name in enumerate( sorted(glob("*.tiff"),key=len) ):
        img = Image.open(file_name)
        half_the_width = img.size[0] / 2
        half_the_height = img.size[1] / 2
        img3 = img.crop(
        (
           half_the_width - 50,
           half_the_height - 60,
           half_the_width + 50,
           half_the_height + 90
        )
        )
        img3.save("jaffecrop//%s.tiff"%file_name)	
        i=i+1

