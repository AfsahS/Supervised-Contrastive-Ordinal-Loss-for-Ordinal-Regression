import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
sys.modules['Image'] = Image
from skimage.transform import resize
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator

class DCMDataFrameIterator(DataFrameIterator):
    def __init__(self, *arg, **kwargs):
        self.white_list_formats = ('dcm')
        super(DCMDataFrameIterator, self).__init__(*arg, **kwargs)
        self.dataframe = kwargs['dataframe']
        self.x = self.dataframe[kwargs['x_col']]
        self.y = self.dataframe[kwargs['y_col']]
        self.color_mode = kwargs['color_mode']
        self.target_size = kwargs['target_size']

    def _get_batches_of_transformed_samples(self, indices_array):
        # get batch of images
        batch_x = np.array([self.read_dcm_as_array(dcm_path, self.target_size, color_mode=self.color_mode)
                            for dcm_path in self.x.iloc[indices_array]])

        batch_y = np.array(self.y.iloc[indices_array].astype('float32'))  # astype because y was passed as str

        # transform images
        if self.image_data_generator is not None:
            for i, (x, y) in enumerate(zip(batch_x, batch_y)):
                transform_params = self.image_data_generator.get_random_transform(x.shape)
                batch_x[i] = self.image_data_generator.apply_transform(x, transform_params)
#
#         return [batch_x,batch_x], batch_y ### for stage 2
        return batch_x, batch_y  #### for stage 1


    @staticmethod
    def read_dcm_as_array(dcm_path, target_size=(300, 300), color_mode='grayscale'):
        im = pydicom.dcmread(dcm_path).pixel_array

        image_array = np.double(im)
        out = np.zeros(image_array.shape, np.double)

        image_array = resize(image_array,target_size,preserve_range=True).astype('float32')

        image_array = cv2.normalize(image_array, out, 1.0, 0.0, cv2.NORM_MINMAX).astype('float32')
        img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        if color_mode == 'rgb':
             image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        return image_array