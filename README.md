# Landsat8Tools
Some tools to read, write, radiometric calibration and cut a image to tiles

# how to use
"""
just like following:
tool = Landsat8Tools()
mtl_file_path = "your path"
save_folder = "your save folder"
"""

now you can read all kinds of bands

"""

multi_image, multi_projection, multi_geotransform = tool.read_multi_band_to_image(mtl_file_path)
pan_image, pan_projection, pan_geotransform = tool.read_pan_band_to_image(mtl_file_path)
thermal_image, thermal_projection, thermal_geotransform = tool.read_thermal_band_to_image(mtl_file_path)

"""
and you can do radiometric calibration

"""

tool.radiometric_calibration(mtl_file_path, save_folder, cali_type = 'radiance')

"""

you can cut a image to tiles

"""

tool.cut_to_tile_with_geoinfo(save_folder, (256, 256, 1), pan, pan_projection, pan_geotransform)

"""
