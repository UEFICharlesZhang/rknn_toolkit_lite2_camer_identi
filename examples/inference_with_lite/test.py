import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite

# decice tree for rk356x/rk3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

def get_host():
    # get platform and device type
    system = platform.system()
    machine = platform.machine()
    os_machine = system + '-' + machine
    if os_machine == 'Linux-aarch64':
        try:
            with open(DEVICE_COMPATIBLE_NODE) as f:
                device_compatible_str = f.read()
                if 'rk3588' in device_compatible_str:
                    host = 'RK3588'
                elif 'rk3562' in device_compatible_str:
                    host = 'RK3562'
                else:
                    host = 'RK3566_RK3568'
        except IOError:
            print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
            exit(-1)
    else:
        host = os_machine
    return host

INPUT_SIZE = 224

RK3566_RK3568_RKNN_MODEL = 'resnet18_for_rk3566_rk3568.rknn'
RK3588_RKNN_MODEL = 'resnet18_for_rk3588.rknn'
RK3562_RKNN_MODEL = 'resnet18_for_rk3562.rknn'

def capture_pic():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port) 
  
    # reading the input using the camera 
    result, image = cam.read() 
    
    # If image will detected without any error,  
    # show result 
    if result: 
    
        # showing result, it take frame name and image  
        # output 
        # cv2.imshow("pic_capture", image) 
    
        # saving image in local storage 
        # cv2.imshow("org     image", image)
        # cv2.waitKey(0) 

        dim = (224, 224)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

        # cv2.imshow("Resized image", image)
        # cv2.waitKey(0) 

        cv2.imwrite("pic_capture.png", image) 
        # cv2.resize("pic_capture.png",224,224)

def show_name(index):
    filehandle = open("imagenet_classes_cn.en.zh-CN.txt","r")
    listoflines = filehandle.readlines()
    filehandle.close()
    # print(listoflines[index])
    # remove return
    out_str="{}".format(listoflines[index].strip("\r\n"))
    # print(out_str)
    return out_str

def show_top5(result):
    output = result[0].reshape(-1)
    # softmax
    output = np.exp(output)/sum(np.exp(output))
    output_sorted = sorted(output, reverse=True)

    top5_str = 'resnet18\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                tempstr1 = '{}'.format(index[j])
                # remove []
                temp2 = format(tempstr1.strip("[]"))
                # print(temp2)
                #split str incase "[123 456]"
                temp3 = temp2.split()
                nameindex = int(temp3[0])
                # print(show_name(nameindex))
                topi = '{}: {}\n'.format(show_name(nameindex), value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


if __name__ == '__main__':

    host_name = get_host()
    if host_name == 'RK3566_RK3568':
        rknn_model = RK3566_RK3568_RKNN_MODEL
    elif host_name == 'RK3562':
        rknn_model = RK3562_RKNN_MODEL
    elif host_name == 'RK3588':
        rknn_model = RK3588_RKNN_MODEL
    else:
        print("This demo cannot run on the current platform: {}".format(host_name))
        exit(-1)

    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    capture_pic()
    ori_img = cv2.imread('./pic_capture.png')
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    # run on RK356x/RK3588 with Debian OS, do not need specify target.
    if host_name == 'RK3588':
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img])
    show_top5(outputs)
    print('done')

    rknn_lite.release()
