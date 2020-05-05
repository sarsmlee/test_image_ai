# -*- coding: utf-8 -*-
from imageai.Prediction import ImagePrediction
import tensorflow as tf
import time

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    pass


class PredictImage(object):
    def __init__(self):
        self.image_path = ''
        self.model_path = r'...'  # add training files
        print(self.model_path)
        self.data_dict = dict()

    def change_path(self, input_path):
        self.image_path = input_path

    def func_model(self):
        prediction = ImagePrediction()
        prediction.setModelTypeAsDenseNet()
        prediction.setModelPath(self.model_path)
        prediction.loadModel()

        time_init = time.time()

        predictions, probabilities = prediction.predictImage(self.image_path, result_count=5)
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            self.data_dict[eachPrediction] = eachProbability
            print(eachPrediction + " : " + str(eachProbability))

        time2 = time.time()
        print('this is：' + max(self.data_dict, key=self.data_dict.get))
        print('time：' + str(time2 - time_init) + 's')


if __name__ == '__main__':
    pass

