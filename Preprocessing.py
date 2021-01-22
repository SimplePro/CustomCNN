from keras.preprocessing import image
import cv2
import numpy as np
import os

from keras_preprocessing.image import ImageDataGenerator

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
from urllib.request import urlretrieve


class ImagePreprocess:

    def __init__(self):
        self.trainDataGen = ImageDataGenerator(rescale=1. / 255, rotation_range=30, width_shift_range=0.1,
                                          height_shift_range=0.1,
                                          shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True,
                                          fill_mode='nearest')

        self.testDataGen = ImageDataGenerator(rescale=1. / 255)

    def img_to_pred(self, img=None):
        if img is None:
            raise Exception("img must not be None")

        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def resize_img(self, directory=None, file=None, new_name=False, name_list=None, size=None):
        if directory is None and file is None:
            raise Exception("directory and file must not be None")
        if size is None:
            raise Exception("size must not be None")
        if new_name and name_list is None:
            raise Exception("name_list must not be None when new_name is True")

        if not new_name:
            if not directory is None:
                file_list = [file for file in os.listdir(directory) if ".jpg" in file or ".png" in file or ".jpeg" in file]
                for file in file_list:
                    img = cv2.imread(directory + file)
                    img = cv2.resize(img, dsize=size)
                    cv2.imwrite(directory + file, img)

            if not file is None:
                img = cv2.imread(file)
                img = cv2.resize(img, dsize=size)
                cv2.imwrite(file, img)

        if new_name:
            file_list = [file for file in os.listdir(directory) if ".jpg" in file or ".png" in file or ".jpeg" in file]
            for i, file in enumerate(file_list):
                img = cv2.imread(directory + file)
                img = cv2.resize(img, dsize=size)
                cv2.imwrite(directory + name_list[i], img)

    def search(self, search_name=None, directory=None):
        if search_name is None:
            raise Exception("search_name must not be None")
        if directory is None:
            raise Exception("directory must not be None")

        driver = webdriver.Chrome()
        driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&ogbl")
        elem = driver.find_element_by_name("q")
        elem.send_keys(search_name)
        elem.send_keys(Keys.RETURN)

        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    driver.find_element_by_css_Selector(".mye4qd").click()
                except:
                    break
            last_height = new_height

        images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
        count = 1
        for image in images:
            try:
                image.click()
                time.sleep(0.2)
                imgURL = driver.find_element_by_xpath(
                    '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute("src")
                urlretrieve(imgURL, directory + search_name + str(count) + ".jpg")
                count += 1
            except:
                pass

        driver.close()