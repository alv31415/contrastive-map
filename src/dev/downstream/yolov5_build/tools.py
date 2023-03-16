
import paddle
import paddle.nn as nn
from models import YOLOX
import numpy as np
from utils.boxes import postprocess, preproc
from utils.visualize import vis
from utils.utils_bbox import non_max_suppression
from pd_model.x2paddle_code import ONNXModel
import cv2
from PIL import Image
import torch


def decode_outputs1(hw,strides1,outputs,input_shape, dtype):

    grids = []
    strides = []
    for (hsize, wsize), stride in zip(hw, strides1):
        yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
        grid = paddle.stack((xv, yv), 2).reshape([1, -1, 2])
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(paddle.full((*shape, 1), stride))
        #print("strides", len(strides))



    grids = paddle.concat(grids, axis=1).astype(dtype)
    strides = paddle.concat(strides, axis=1).astype(dtype)

    outputs[:, :, :2] = (outputs[:, :, :2] + grids) * strides
    outputs[:, :, 2:4] = paddle.exp(outputs[:, :, 2:4]) * strides

    #-----------------#
    #   归一化
    #-----------------#
    outputs[:, :,0] = outputs[:, :,0]  / input_shape[1]
    outputs[:, :,2] = outputs[:, :,2] / input_shape[1]
    outputs[:, :, 1] = outputs[:, :, 1] / input_shape[0]
    outputs[:, :, 3] = outputs[:, :, 3] / input_shape[0]
    return outputs


def preprocimg(image,size):
    iw, ih  = image.size
    w, h    = size
    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)
    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    image=np.array(new_image, dtype='float32')


    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])

    imgs = np.transpose(image,(2, 0, 1))
    imgs1 = paddle.to_tensor(imgs[None, :, :, :])
    return imgs1



#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)
def get_center_coor(x1,y1,x2,y2):
    return (x2-x1)/2+x1,(y2-y1)/2+y1


def model_det(imgs,model,input_shape):
    strides = [8, 16, 32]
    outputs = model(imgs)
    hw = [x.shape[-2:] for x in outputs]
    outputs = paddle.concat([x.flatten(start_axis=2) for x in outputs], axis=2).transpose([0, 2, 1])
    outputs[:, :, 4:] = paddle.nn.functional.sigmoid(outputs[:, :, 4:])
    outputs = decode_outputs1(hw, strides, outputs, input_shape, dtype='float32')
    outputs = torch.from_numpy(outputs.numpy())
    return outputs

def plot_image(frame,results,Area,class_names,size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    top_label = np.array(results[0][:, 6], dtype='int32')
    top_conf = results[0][:, 4] * results[0][:, 5]
    top_boxes = results[0][:, :4]

    # ---------------------------------------------------------#
    #   图像绘制
    # ---------------------------------------------------------#
    for i, c in list(enumerate(top_label)):
        predicted_class = class_names[int(c)]
        box = top_boxes[i]
        score = top_conf[i]
        top, left, bottom, right = box
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(size[1], np.floor(bottom).astype('int32'))
        right = min(size[0], np.floor(right).astype('int32'))

        print(predicted_class, top, left, bottom, right)    #b'person 0.87' 345 1104 610 1278

        cv2.rectangle(frame, (left,top), (right,bottom), (255,0,0), 2)   #  b'person 0.87' 634 821 720 998

        x1,y1=get_center_coor(left, top, right,bottom)
        if x1<Area[2] and x1>Area[0] and y1>Area[1] and y1<Area[3]:
            cv2.putText(frame, 'Warning!intruder!!', (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            print("有人闯入")
                                                          #x1, y1, x2, y2=Area
    return frame