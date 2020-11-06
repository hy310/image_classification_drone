import sys
import cv2
#import os
import threading
import traceback
import time
#from PIL import Image
import numpy as np
#from keras.preprocessing.image import load_img
try:
    from openvino.inference_engine import IENetwork, IEPlugin
    from openvino import inference_engine as ie
except Exception as e:
    exception_type = type(e).__name__
    print("The following error happened while importing Python API module:\n[ {} ] {}".format(exception_type, e))
    sys.exit(1)
class ObjectClassifier_Cifar10():
    def __init__(self, device, getFrameFunc):
        self.getFrameFunc = getFrameFunc
        self.originFrame = None
        self.processedFrame = None
############### 모델이름 Please write mobilenet imageclassifcation xml and bin file############## 
        model_xml = './models/mobilenetv2.xml'
        model_bin = './models/mobilenetv2.bin'

        cpu_extension = None# '/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so'
############### 레이블이름 Please write label fiel ################################################ 
        with open('./models/mobilenetv2.txt', 'rt',encoding='utf-8') as f:
            lines = f.readlines()
        self.labels = list(map(lambda x: x.replace('\n', ''), lines))

        net = IENetwork(model=model_xml, weights=model_bin)
        assert len(net.inputs.keys()) == 1
        assert len(net.outputs) == 1

        print(device)
        plugin = IEPlugin(device=device)
        print(IEPlugin)
        if cpu_extension and 'CPU' in device:
            plugin.add_cpu_extension(cpu_extension)

        self.exec_net = plugin.load(network=net)
        #del net

        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        print("Loading IR to the plugin...")
        n, c, self.h, self.w = net.inputs[self.input_blob].shape #?? 겠다
        print(n, c, self.h, self.w)

        self.sortedClassifiedList = []
        self.infer_time = 0
        self.inferFPS = 15

        processThread = threading.Thread(target=self.inferenceThread)
        processThread.daemon = True
        processThread.start()

    def pre_process_image(self, image, img_height=224):
        # Model input format
        n, c, h, w = [1, 3, img_height, img_height]
        processedImg = cv2.resize(image, (h, w), interpolation=cv2.INTER_AREA)

        # Normalize to keep data between 0 - 1
        processedImg = (np.array(processedImg) - 0) / 255.0

        # Change data layout from HWC to CHW
        processedImg = processedImg.transpose((2, 0, 1))
        processedImg = processedImg.reshape((n, c, h, w))
        return image, processedImg

    def detect(self):
        image, processedImg = self.pre_process_image(self.originFrame)

        infer_start = time.time()
        res = self.exec_net.infer(inputs={self.input_blob: processedImg})
        self.infer_time = time.time() - infer_start

        output_node_name = list(res.keys())[0]
        res = res[output_node_name]

        # Predicted class index.
        sortedIdx = np.argsort(res[0])[::-1]

        #index_max = max(range(len(res[0])), key=res[0].__getitem__)

        self.sortedClassifiedList.clear()
        #sortedList = sorted(range(len(res[0])), key=lambda i: res[0][i], reverse=True)

        for idx in sortedIdx:
            self.sortedClassifiedList.append((idx, res[0][idx]))

    def inferenceThread(self):
        while True:
            frame = self.getFrameFunc()
            if frame is not None:
                try:
                    self.originFrame = frame.copy()
                    self.detect()
                    time.sleep(1.0/self.inferFPS)

                except Exception as error:
                    print(error)
                    traceback.print_exc()
                    print("catch error")
    def getProcessedData(self):
        return self.infer_time, self.sortedClassifiedList
    def setInferFPS(self, newFPS):
        self.inferFPS = newFPS


if __name__ == "__main__":
    
    frame = None
    def getOriginFrame():
        return frame
    

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = capture.read()

    classifier = ObjectClassifier_Cifar10('CPU',getOriginFrame)
    while capture.isOpened():
        ret, frame = capture.read()
        data = classifier.getProcessedData()
