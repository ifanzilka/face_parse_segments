#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2

def from_numpy(image, mean =(0.485, 0.456, 0.406)  , std = (0.229, 0.224, 0.225)):
    image = np.array(image).transpose(1, 2, 0)  #(2 0 1) -> 0 1 2    480 640 3 -> 3 480 640 ->
    image = image * std
    image = image + mean
    image = image * 255
    return image

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, image_path = "tmp", save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
   
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    #vis_im = im.copy()
    #vis_im = from_numpy(vis_im)
    
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8) # array color 3
    #print(vis_parsing_anno_color.shape)
    #print(vis_parsing_anno_color)
    
    rgb_img = cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR) 
    vis_im = cv2.addWeighted(rgb_img, 0.4, vis_parsing_anno_color, 0.6, 0)
    
    file_name = save_path.split("/")
    #return
    # Save result or not
    if save_im:
        cv2.imwrite( file_name[0] + '/' + "1_" + file_name[1], vis_parsing_anno)
        cv2.imwrite( file_name[0] + '/' + "2_" + file_name[1] , vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

import onnxruntime
import onnx

def evaluate(respth='new_image', dspth='image/', cp="onnnx_files/onnx_model_name.onnx"):

    if not os.path.exists(respth):
        os.makedirs(respth)

    ### Cheks ###
    onnx_model = onnx.load(cp)
    onnx.checker.check_model(onnx_model)
    
    ort_session = onnxruntime.InferenceSession(cp)

    # Вывод что из себя представляет
    print(ort_session.get_inputs())
    tmp = [i.name for i in ort_session.get_inputs()]
    print(tmp)
    
    def to_numpy(image, mean =(0.485, 0.456, 0.406)  , std = (0.229, 0.224, 0.225)):
        image = np.array(image)
        image = image / 255
        image = (image - mean ) / std
        image = np.array(image).transpose(2, 0, 1)
        return image


    # to_tensor = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    
    for image_path in os.listdir(dspth):
        img_orig  = Image.open(osp.join(dspth, image_path))
        img_orig = img_orig.resize((480, 640), Image.BILINEAR)

        img = img_orig.copy()
        img = to_numpy(img)

        
        img = np.array([img for _ in range(16)])
        
        # Словарь
        ort_inputs = {ort_session.get_inputs()[0].name: img.astype(np.float32)}
        
        ort_outs = ort_session.run(None, ort_inputs)[0]
        
        # Массив вероятностей -> в массив сегментов
        parsing = ort_outs[0].argmax(0)

        print(np.unique(parsing))

        vis_parsing_maps(img_orig, parsing, stride=1, save_im=True, image_path = image_path, save_path=osp.join(respth, image_path))

if __name__ == "__main__":
    evaluate(dspth='image/', cp='onnnx_files/onnx_model_name.onnx')