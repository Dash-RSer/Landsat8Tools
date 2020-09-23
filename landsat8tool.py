# -*-coding:utf-8-*-
#
# @author:rsstudent
#

import numpy as np
from osgeo import gdal
from osgeo import gdal_array
import cv2 as cv
import math
from numba import jit
from functools import wraps
import os

class Landsat8Tools(object):

    def __init__(self):
        self.LANDSAT8_MULTI_BAND_NUM = 7
        self.LANDSAT8_THERMAL_BAND_NUM = 2
        self.LANDSAT8_PAN_BAND_NUM = 1
        self.LANDSAT8_PAN_BAND = 8
        self.LANDSAT8_THERMAL_START_BAND = 10
        self.multi_nan_position = []
        self.pan_nan_position = []
        self.thermal_nan_position = []
        self.process_image_nan_position = []


    def __get_base_name(self, full_name):
        base_names = full_name.split("_")[:-1]
        base_name = ""
        for string in base_names:
            base_name += (string + "_")
        return base_name


    def read_multi_band_to_image(self, mtl_file_path):
        
        multi_band_file_names = []

        base_name = self.__get_base_name(mtl_file_path)
        
        for band in range(self.LANDSAT8_MULTI_BAND_NUM):
            band_name = base_name + "B" + str(band + 1) + ".tif"
            multi_band_file_names.append(band_name)

        if len(multi_band_file_names) == 0:
            raise Exception("The filename is incorrect!")

        dataset = gdal.Open(multi_band_file_names[0], gdal.GA_ReadOnly)

        if dataset == None:
            raise Exception("Fail to open the image file.")
        
        height = dataset.RasterYSize
        width = dataset.RasterXSize

        image = np.zeros((height, width, self.LANDSAT8_MULTI_BAND_NUM), dtype = np.float16)
        projection = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        del dataset

        print("read multi band start, please wait...")
        for band in range(self.LANDSAT8_MULTI_BAND_NUM):
            ds = gdal.Open(multi_band_file_names[band], gdal.GA_ReadOnly)
            band_image = ds.GetRasterBand(1)
            image[:, :, band] = band_image.ReadAsArray()
            del ds
        print("read  multi band finish.")
        self.multi_nan_position = np.where(image == 0)
        image[self.multi_nan_position] = np.nan

        return image, projection, geotransform

    def read_pan_band_to_image(self, mtl_file_path):
        base_name = self.__get_base_name(mtl_file_path)
        pan_file_name = base_name + "B8" + ".tif"
        
        dataset = gdal.Open(pan_file_name, gdal.GA_ReadOnly)
        if dataset == None:
            raise Exception("Image name error.")
        else:
            datatype = np.float16
            height = dataset.RasterYSize
            width = dataset.RasterXSize
            projection = dataset.GetProjection()
            geotransform = dataset.GetGeoTransform()

            pan_image = np.zeros((height, width, 1), dtype = datatype)
            print("read pan start, please wait...")
            band_data = dataset.GetRasterBand(1)
            pan_image[:, :, 0] = band_data.ReadAsArray()
            del dataset

        self.pan_nan_position = np.where(pan_image == 0)
        pan_image[self.pan_nan_position] = np.nan
        print("read pan finish.")

        return pan_image, projection, geotransform

    def read_thermal_band_to_image(self, mtl_file_path):
        nan_position = []
        base_name = self.__get_base_name(mtl_file_path)
        thermal_band10_file_name = base_name + "B10" + ".tif"
        thermal_band11_file_name = base_name + "B11" + ".tif"

        names = []
        names.append(thermal_band10_file_name)
        names.append(thermal_band11_file_name)

        dataset = gdal.Open(thermal_band10_file_name, gdal.GA_ReadOnly)
        if dataset == None:
            raise Exception("Image name error.")
        else:
            datatype = np.float16
            height = dataset.RasterYSize
            width = dataset.RasterXSize
            projection = dataset.GetProjection()
            geotransform = dataset.GetGeoTransform()
            thermal_image = np.zeros((height, width, 2), dtype = datatype)
            del dataset
        print("read thermal band start, please wait...")
        for band in range(self.LANDSAT8_THERMAL_BAND_NUM):
            ds = gdal.Open(names[band], gdal.GA_ReadOnly)
            band_data = ds.GetRasterBand(1)
            thermal_image[:, :, band] = band_data.ReadAsArray()
            del ds
        
        self.thermal_nan_position = np.where(thermal_image == 0)
        thermal_image[self.thermal_nan_position] == np.nan
        print("read thermal band finish.", end = "")

        return thermal_image, projection, geotransform

    def read_single_band_to_image(self, mtl_file_path, band_num):

        base_name =self.__get_base_name(mtl_file_path)
        nan_position = []
        fullbandname = base_name + "B" + str(band_num) + ".tif"
        dataset = gdal.Open(fullbandname, gdal.GA_ReadOnly)
        if dataset == None:
            raise Exception("Image name error.")
        else:
            datatype = np.float16
            height = dataset.RasterYSize
            width = dataset.RasterXSize
            projection = dataset.GetProjection()
            geotransform = dataset.GetGeoTransform()

            band_image = np.zeros((height, width, 1), dtype = datatype)

            print("read single band start")
            band_data = dataset.GetRasterBand(1)
            band_image[:, :, 0] = band_data.ReadAsArray()
            del dataset
            print("read single band finish.")
        
        self.process_image_nan_position = np.where(band_image == 0)
        band_image[self.process_image_nan_position] == np.nan

        return band_image, projection, geotransform

    def save(self, save_path, image, projection, geotransform, format = 'GTiff'):
        datatype = gdal.GDT_Float32
        DIMENSION_OF_IMAGE = 3
        if len(image.shape) != DIMENSION_OF_IMAGE:
            raise Exception("The dimension of the image is incorrect.")
        else:
            height = image.shape[0]
            width = image.shape[1]
            channels = image.shape[2]

        driver = gdal.GetDriverByName(format)
        ds_to_save = driver.Create(save_path, width, height, channels, datatype)
        ds_to_save.SetGeoTransform(geotransform)
        ds_to_save.SetProjection(projection)

        print("save tool start, please wait...")
        for band in range(channels):
            ds_to_save.GetRasterBand(band + 1).WriteArray(image[:, :, band])
            ds_to_save.FlushCache()

        print("save finish.")
        del image
        del ds_to_save

    def radiometric_calibration(self, mtl_file_path, save_folder, cali_type = "radiance"):
        
        f = open(mtl_file_path, 'r')
        metadata = f.readlines()
        f.close()
        radiance_multi_paras = []
        radiance_add_paras = []

        reflectance_multi_paras = []
        reflectance_add_paras = []
        
        radiance_paras_start_line = 0
        reflectance_paras_start_line = 0

        for lines in metadata:
            test_line = lines.split("=")
            if test_line[0] == '    REFLECTANCE_MULT_BAND_1 ':
                break
            else:
                reflectance_paras_start_line += 1

        for lines in range(reflectance_paras_start_line, reflectance_paras_start_line + 9):
            parameter = float(metadata[lines].split('=')[1])
            reflectance_multi_paras.append(parameter)

        for lines in range(reflectance_paras_start_line + 9, reflectance_paras_start_line + 18):
            parameter = float(metadata[lines].split('=')[1])
            reflectance_add_paras.append(parameter)        

        for lines in metadata:
            test_line = lines.split("=")
            if test_line[0] == '    RADIANCE_MULT_BAND_1 ':
                break
            else:
                radiance_paras_start_line += 1

        for lines in range(radiance_paras_start_line, radiance_paras_start_line + 11):
            parameter = float(metadata[lines].split('=')[1])
            radiance_multi_paras.append(parameter)

        for lines in range(radiance_paras_start_line + 11, radiance_paras_start_line + 22):
            parameter = float(metadata[lines].split('=')[1])
            radiance_add_paras.append(parameter)

        if cali_type != "radiance" and cali_type != "reflectance":
            raise Exception("cali_type is incorrect.")

        if cali_type == "radiance":
            
            multi_image, multi_projection, multi_geotransform = self.read_multi_band_to_image(mtl_file_path)

            print("radiance radiomereic calibration start.")

            for band in range(self.LANDSAT8_MULTI_BAND_NUM):
                gain = radiance_multi_paras[band]
                offset = radiance_multi_paras[band]
                multi_image[:, :, band] = multi_image[:, :, band]*gain + offset
            
            multi_image[self.multi_nan_position] = np.nan
            multi_image_name = "radiance_multi.tif"
            save_path = os.path.join(save_folder, multi_image_name)
            self.save(save_path, multi_image, multi_projection, multi_geotransform)
            del multi_image

            pan_image, pan_projection, pan_geotransform = self.read_pan_band_to_image(mtl_file_path)
            pan_gain = radiance_multi_paras[7]
            pan_offset = radiance_add_paras[7]

            pan_image = pan_image * gain + offset
            pan_image[self.pan_nan_position] = np.nan
            pan_image_name = "radiance_pan.tif"
            save_path = os.path.join(save_folder, pan_image_name)
            self.save(save_path, pan_image, pan_projection, pan_geotransform)
            del pan_image
            
            thermal_image, thermal_projection, thermal_geotransform = self.read_thermal_band_to_image(mtl_file_path)
            thermal_multi_paras = []
            thermal_add_paras = []

            thermal_multi_paras.append(radiance_multi_paras[9])
            thermal_multi_paras.append(radiance_multi_paras[10])
            thermal_add_paras.append(radiance_add_paras[9])
            thermal_add_paras.append(radiance_add_paras[10])
            
            for band in range(self.LANDSAT8_THERMAL_BAND_NUM):
                thermal_image[:, :, band] = thermal_image[:, :, band] * thermal_multi_paras[band] + thermal_add_paras[band]
    
            thermal_image[self.thermal_nan_position] = np.nan
            thermal_image_name = "radiance_thermal.tif"
            save_path = os.path.join(save_folder, thermal_image_name)
            self.save(save_path, thermal_image, thermal_projection, thermal_geotransform)
            
            print("ridiometric calibraion finish.")
            del thermal_image
        else:
            
            multi_image, multi_projection, multi_geotransform = self.read_multi_band_to_image(mtl_file_path)

            print("reflectance radiomereic calibration start.")
            for band in range(self.LANDSAT8_MULTI_BAND_NUM):
                gain = reflectance_multi_paras[band]
                offset = reflectance_multi_paras[band]
                multi_image[:, :, band] = multi_image[:, :, band]*gain + offset

            multi_image[self.multi_nan_position] = np.nan
            multi_image_name = "reflectance_multi.tif"
            save_path = os.path.join(save_folder, multi_image_name)
            self.save(save_path, multi_image, multi_projection, multi_geotransform)
            del multi_image

            pan_image, pan_projection, pan_geotransform = self.read_pan_band_to_image(mtl_file_path)
            pan_gain = reflectance_multi_paras[7]
            pan_offset = reflectance_add_paras[7]

            pan_image = pan_image * gain + offset
            pan_image[self.pan_nan_position] = np.nan

            pan_image_name = "reflectance_pan.tif"
            save_path = os.path.join(save_folder, pan_image_name)
            self.save(save_path, pan_image, pan_projection, pan_geotransform)
            del pan_image
            
            thermal_image, thermal_projection, thermal_geotransform = self.read_thermal_band_to_image(mtl_file_path)
            thermal_multi_paras = []
            thermal_add_paras = []

            thermal_multi_paras.append(radiance_multi_paras[9])
            thermal_multi_paras.append(radiance_multi_paras[10])
            thermal_add_paras.append(radiance_add_paras[9])
            thermal_add_paras.append(radiance_add_paras[10])
            
            
            for band in range(self.LANDSAT8_THERMAL_BAND_NUM):
                thermal_image[:, :, band] = thermal_image[:, :, band] * thermal_multi_paras[band] + thermal_add_paras[band]

            thermal_image[self.thermal_nan_position] = np.nan
            thermal_image_name = "radiance_thermal.tif"
            save_path = os.path.join(save_folder, thermal_image_name)
            self.save(save_path, thermal_image, thermal_projection, thermal_geotransform)
            print("radiometric calibraion finish.")
            del thermal_image

    def cut_to_tile_with_geoinfo(self, save_folder, image_size, image, projection, geotransform, format = "GTiff"):
        print("cut to tile with geoinfo tool start.")
        DIMENSION_OF_IMAGE = 3

        if len(image_size) !=  DIMENSION_OF_IMAGE:
            raise Exception("your image dimension is incorrect.Need 3-d array.")

        image_height = image.shape[0]
        image_width = image.shape[1]
        image_channels = image.shape[2]

        tile_height = image_size[0]
        tile_width = image_size[1]
        channels = image_size[2]

        if image_channels != channels:
            raise Exception("Your image channels isn't match the required channels.")

        num_rows = math.floor(image_height/tile_height)
        num_cols = math.floor(image_width/tile_width)
        useful_height = num_rows * tile_height
        useful_width = num_cols * tile_width

        image = np.array(image[:useful_height, :useful_width, :])

        x_origin_point = geotransform[0]
        x_pixel_size = geotransform[1]
        x_ro = geotransform[2]
        y_origin_point = geotransform[3]
        y_ro = geotransform[4]
        y_pixel_size = geotransform[5]
        i = 0
        x_offset = 0
        y_offset = 0
        
        for tile_row in range(num_rows):
            for tile_col in range(num_cols):
                tile = image[tile_row*tile_width: (tile_row+1)*tile_width, tile_col*tile_height:(tile_col+1)*tile_height, :]
                filename = str(i) + '.tif'
                path = os.path.join(save_folder, filename)
                tile = np.float32(tile)
                tile_x_origin_point = x_origin_point + x_pixel_size * tile_col * tile_width
                tile_y_origin_point = y_origin_point + y_pixel_size * tile_row * tile_height
                tile_geotransform = (tile_x_origin_point, x_pixel_size, x_ro, tile_y_origin_point, y_ro, y_pixel_size)
                self.save(path, tile, projection, tile_geotransform, format)
                i += 1
                del tile

        print("cut to tile with geoinfo tool finish.")
        del image

if __name__ == "__main__":
    tool = Landsat8Tools()
    mtl_file_path = "F:\SRGAN_program\dataset\LC81290352019095LGN00\LC08_L1TP_129035_20190405_20190422_01_T1_MTL.txt"
    save_folder = "F:\\SRGAN_program\\dataset\\LC81290352019095LGN00\\tiles"
    save_folder1 = "F:\\SRGAN_program\\dataset\\LC81290352019095LGN00\\test\\ppan"
    pan, projection, geotransform = tool.read_pan_band_to_image(mtl_file_path)
    tool.cut_to_tile_with_geoinfo(save_folder, (256, 256, 1), pan, projection, geotransform)
    #tool.save(save_folder1, pan, projection, geotransform)
        