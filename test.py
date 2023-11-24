# from PaddleOCR import PaddleOCR

# ocr = PaddleOCR(use_gpu=True)
# # ocr.summary()
# import paddle

# paddle.Model(ocr).summary()

from paddle.inference import create_predictor
from paddle.fluid.libpaddle import PaddleInferPredictor


from paddle.fluid import libpaddle
paddle.fluid.io.save_inference_model("/home/test/")


