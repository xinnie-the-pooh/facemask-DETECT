# fanhuazeng at sdp compile and design ver3.4 （final）
# 2021-7 zengfanhua@live.com 
############################################
# 致谢前，我是要骂的，骂这个傻逼山东职业学院，做东西不给报销还要贴钱，，题外话是我2020年计算机安全这个课竟然要补考，不过五分钟写完了就过了。
# 神奇的谢某人在网课时期将计算机讲的一塌糊涂。
# 后来考计算机三级把名字搞成z学校了，，，证还下来了，，离谱
# 傻逼郭泽岩拿这玩意获了8000块的奖金，我一分没有，所以去死吧 郭泽岩的qq号1873608401，z小子，其次我那这玩意搞了个软著，结果代理那边用c++给我截了一段弄上去了。。300快也没啥用
##############################################################
# 特别感谢国际友人chandrikadeb7 的Face-Mask-Detection项目以及回复，神速回复，geek精神的体现啊
# 特别感谢c.z创始人崔玉清先生于我小学时的鼓励（看来还是不叫z.c啊），及嘉宁兄的赏识。（大好人王嘉宁）
###########################################################################################################
# also thanks for my past love -- chunyuhan ，wether she remeber me or not.  i owe her too much 。if god give me a chance to back again ，i will accompany her forever。
# 若有来世，幸福为与你相伴
#########################################################################################################################
# 感谢我的高中济南中学，是他给了我们逐梦的机会，致谢所有2015级，2016级，2017级同学，尤其是孔今旭和他的彩虹桥的朋友们 世界未来因你而精彩 
# 感谢山东职业学院2018级电气系同学，以及蔷薇院李某飞同志等，原电气史劲帅同志，以及dr.gong,和他的好兄弟强某，，and my davalishi lison xu。
# my erver hero Stephen Gary Wozniak！！ 
#####################################################################
# 那一年我与她一起看了电影《君の名は》。但最终现实中我不并是主角，或许更像《大鱼海棠》的结局，真实而深刻。
############################################################################################################
# 尽管许多事情想不明白，但还是放手去做吧。
# 时光不老，我们不散~






#程序运行需要好多的库，不一一列举，自行对照，可以使用bat批处理来查看错误一一查找。
#之前版本的注释基本都没怎么留，残留的代码也较少，结构较简单




import cv2
import argparse
import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from PIL import Image, ImageDraw, ImageFont
# import win32com.client
# speak = win32com.client.Dispatch('SAPI.SPVOICE')
import time
import threading
import os
from pygame import mixer
mixer.init()
sound = mixer.Sound('mask.wav')

sound_cooldown = False


# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)
# sound = mixer.Sound('Alert_Beep.wav')
id2class = {0: 'Mask', 1: 'NoMask'}
id2chiclass = {0: '您戴了口罩', 1: '您没有戴口罩'}
colors = ((0, 255, 0), (255, 0 , 0))

def puttext_chinese(img, text, point, color):
    pilimg = Image.fromarray(img)
    draw = ImageDraw.Draw(pilimg)  # 图片上打印汉字
    fontsize = int(min(img.shape[:2])*0.04)
    font = ImageFont.truetype("simhei.ttf", fontsize, encoding="utf-8")
    y = point[1]-font.getsize(text)[1]
    if y <= font.getsize(text)[1]:
        y = point[1]+font.getsize(text)[1]
    draw.text((point[0], y), text, color, font=font)
    img = np.asarray(pilimg)
    return img
def playSound():
	global sound_cooldown
	if sound_cooldown:return 
	sound.play(time.daylight)
	sound_cooldown = True
	threading.Timer(5, releaseCooldown).start()#线程定时器
def releaseCooldown():
	global sound_cooldown
	sound_cooldown = False
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def inference(net, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True, chinese=True):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
    net.setInput(blob)
    y_bboxes_output, y_cls_output = net.forward(getOutputsNames(net))
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    
    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
    # keep_idxs  = cv2.dnn.NMSBoxes(y_bboxes.tolist(), bbox_max_scores.tolist(), conf_thresh, iou_thresh)[:,0]
    tl = round(0.002 * (height + width) * 0.5) + 1  # line thickness
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        if draw_result:
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[class_id], thickness=tl)
            if chinese:
                image = puttext_chinese(image, id2chiclass[class_id], (xmin, ymin), colors[class_id])  ###puttext_chinese
            else:
                cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_id])
        if class_id==1:#触发条件
            # speak.Speak('请戴口罩')     
             playSound()   
    return image
    
def run_on_video(Net, video_path, conf_thresh=0.5):
    cap = cv2.VideoCapture(video_path)
    cap.set(3,1920)#宽
    cap.set(4,1080)#高
    
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    while status:
        status, img_raw = cap.read()
        if not status:
            print("Done processing !!!")
            break
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_raw = inference(Net, img_raw, target_shape=(260, 260), conf_thresh=conf_thresh)
        cv2.imshow('image', img_raw[:,:,::-1])
        cv2.waitKey(1)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--proto', type=str, default='models/face_mask_detection.prototxt', help='prototxt path')
    parser.add_argument('--model', type=str, default='models/face_mask_detection.caffemodel', help='model path')
    parser.add_argument('--img-mode', type=int, default=0, help='set 1 to run on image, 0 to run on video.')
    parser.add_argument('--img-path', type=str, default='img/demo2.jpg', help='path to your image.')
    parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()
    
    Net = cv2.dnn.readNet(args.model, args.proto)
    if args.img_mode:
        img = cv2.imread(args.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = inference(Net, img, target_shape=(260, 260))
        cv2.namedWindow('detect', cv2.WINDOW_AUTOSIZE)#窗口
        cv2.imshow('detect', result[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        video_path = args.video_path
        if args.video_path == '0':
            video_path = 0
        run_on_video(Net, video_path, conf_thresh=0.5)
