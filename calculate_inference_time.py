import tflite_runtime.interpreter as tflite
import pandas as pd
import cv2
import time
import numpy as np
import os
import glob2

IMAGE_PATH = os.path.abspath('./test_images')
WEIGHT_PATH = os.path.abspath('./weights/chexnet_kaggle_tflite_dynaic_quantization.tflite')
images = sorted(glob2.glob(IMAGE_PATH + '/*.png'))
print(f"Total number of Images : {len(images)}")

def evaluate_timing():
    total_time = 0
    interpreter = tflite.Interpreter(model_path = WEIGHT_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    for idx, img_item in enumerate(images):
        img = cv2.imread(img_item)
        img = cv2.resize(img, (224, 224)) / 255
        test_image = np.expand_dims(img, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details['index'], test_image)
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        total_time += time_taken

    print(f"Total Time taken : {total_time} seconds.")
    print(f"Average Time Taken : {total_time / len(images)} seconds")

if __name__ == "__main__":
    evaluate_timing()

