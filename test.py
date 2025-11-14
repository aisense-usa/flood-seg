from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def get_exif_data(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if not exif_data:
        return None

    gps_data = {}
    for tag_id, value in exif_data.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            for key in value.keys():
                gps_tag = GPSTAGS.get(key, key)
                gps_data[gps_tag] = value[key]
    return gps_data


gps_info = get_exif_data("./data/Preflood.tif")

if gps_info:
    print("✅ GPS Data Found:")
    for k, v in gps_info.items():
        print(f"{k}: {v}")
else:
    print("❌ No GPS data found in this TIFF file.")
