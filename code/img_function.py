# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import debug
import img_math
import img_recognition
import config

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        #初始化SVM实例，设置属性
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat") #如果存在，不再训练，直接导入训练好的结果
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("C:/Users/黄月浩/Desktop/Python_VLPR-master/train/chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)
                    
             #map() 会根据提供的函数对指定序列做映射。
		    #第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
		    #把灰度图的训练样本集合中每个元素逐一送入deskew函数进行抗扭斜处理--也就是把图片摆正
            chars_train = list(map(img_recognition.deskew, chars_train))
            chars_train = img_recognition.preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
           # 保存分类标签到对应的标签集合。
           # 每个字符图片保存在以该字符名称对应的目录中，所以把目录名称对应的ASCII码作为训练数据的分类。
            """
			root:所指的是当前正在遍历的这个文件夹的本身的地址
			dirs:是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
			files:同样是 list , 内容是该文件夹中所有的文件(不包括子目录)

			train\chars2目录下保存有数字和大写字母图片，用于训练
            """
			#os.path.basename(),返回path最后的文件名
			#例：root=train\chars2\7,那么os.path.basename(root)=7
            for root, dirs, files in os.walk("C:/Users/黄月浩/Desktop/Python_VLPR-master/train/charsChinese"):
                if not os.path.basename(root).startswith("zh_"):#目录是单个字母或者数字
                    continue
                pinyin = os.path.basename(root)
                index = img_recognition.provinces.index(pinyin) + PROVINCE_START + 1  # 转化为ASCII字符,1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    #把图片转化为灰度图
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)#训练样本集合
                    # chars_label.append(1)
                    chars_label.append(index)#训练样本标签，这里用字符的ASCII表示
            chars_train = list(map(img_recognition.deskew, chars_train))
            chars_train = img_recognition.preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")

    def img_first_pre(self, car_pic_file):
        """
        :param car_pic_file: 图像文件
        :return:已经处理好的图像文件 原图像文件
        """
        if type(car_pic_file) == type(""):
            img = img_math.img_read(car_pic_file)
        else:
            img = car_pic_file

        pic_hight, pic_width = img.shape[:2]
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        # 缩小图片

        #滤波去噪
        blur = 5
        #(5, 5)表示高斯矩阵的长与宽都是5，标准差取0
        img = cv2.GaussianBlur(img, (blur, blur), 0) 
        oldimg = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 转化成灰度图像

        Matrix = np.ones((20, 20), np.uint8)
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)#将两张相同大小，相同类型的图片融合
        # 创建20*20的元素为1的矩阵 开操作，并和img重合

        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        #Otsu’s二值化（大津法）
        img_edge = cv2.Canny(img_thresh, 100, 200)
        # 找到图像边缘

        Matrix = np.ones((4, 19), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)#闭操作
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)#开操作
        return img_edge2, oldimg
# 识别到的字符、定位的车牌图像、车牌颜色
    def img_color_contours(self, img_contours, oldimg):
        """
        :param img_contours: 预处理好的图像
        :param oldimg: 原图像
        :return: 已经定位好的车牌
        """

        if img_contours.any():
            config.set_name(img_contours)

        pic_hight, pic_width = img_contours.shape[:2]

        card_contours = img_math.img_findContours(img_contours) # 轮廓提取
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)#进行矩形矫正
        colors, car_imgs = img_math.img_color(card_imgs)#颜色判断 缩小边界
        
        predict_result = []
        roi = None
        card_color = None

        for i, color in enumerate(colors):
            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]
                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                #查找水平直方图波峰
                x_histogram = np.sum(gray_img, axis=1)
                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2

                wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                #车牌字符数应大于6
                if len(wave_peaks) <= 6:
                    print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)
                #去除车牌上的分隔点
                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    print("peak less 2:", len(wave_peaks))
                    continue

                part_cards = img_math.seperate_card(gray_img, wave_peaks)
                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉

                    if np.mean(part_card) < 255 / 5:
                        print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])#扩充图像边界
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    part_card = img_recognition.preprocess_hog([part_card])#用于从图片中抽取特征向量
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(resp[0])
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                            continue
                    predict_result.append(charactor)

                roi = card_img
                card_color = color
                break

        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色

    def img_only_color(self, filename, oldimg, img_contours):
        """
        :param filename: 图像文件
        :param oldimg: 原图像文件
        :return: 已经定位好的车牌
        """
        pic_hight, pic_width = img_contours.shape[:2]
        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_yellow, upper_green)
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # 根据阈值找到对应颜色

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        Matrix = np.ones((20, 20), np.uint8)
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)#闭操作
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)#开操作

        card_contours = img_math.img_findContours(img_edge2)#查找图像边缘整体形成的矩形区域
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)#进行矩形矫正
        colors, car_imgs = img_math.img_color(card_imgs)#颜色判断 缩小边界

        predict_result = []
        roi = None
        card_color = None

        for i, color in enumerate(colors):

            if color in ("blue", "yello", "green"):
                card_img = card_imgs[i]

                gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

                # 黄、绿车牌字符比背景暗、与蓝车牌刚好相反，所以黄、绿车牌需要反向
                if color == "green" or color == "yello":
                    gray_img = cv2.bitwise_not(gray_img)
                ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                x_histogram = np.sum(gray_img, axis=1)

                x_min = np.min(x_histogram)
                x_average = np.sum(x_histogram) / x_histogram.shape[0]
                x_threshold = (x_min + x_average) / 2
                wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                if len(wave_peaks) == 0:
                    print("peak less 0:")
                    continue
                # 认为水平方向，最大的波峰为车牌区域
                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                gray_img = gray_img[wave[0]:wave[1]]
                # 查找垂直直方图波峰
                row_num, col_num = gray_img.shape[:2]
                # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
                gray_img = gray_img[1:row_num - 1]
                y_histogram = np.sum(gray_img, axis=0)
                y_min = np.min(y_histogram)
                y_average = np.sum(y_histogram) / y_histogram.shape[0]
                y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半
                wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                if len(wave_peaks) < 6:
                    print("peak less 1:", len(wave_peaks))
                    continue

                wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                max_wave_dis = wave[1] - wave[0]
                # 判断是否是左侧车牌边缘
                if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                    wave_peaks.pop(0)

                # 组合分离汉字
                cur_dis = 0
                for i, wave in enumerate(wave_peaks):
                    if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                        break
                    else:
                        cur_dis += wave[1] - wave[0]
                if i > 0:
                    wave = (wave_peaks[0][0], wave_peaks[i][1])
                    wave_peaks = wave_peaks[i + 1:]
                    wave_peaks.insert(0, wave)

                point = wave_peaks[2]
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)

                if len(wave_peaks) <= 6:
                    print("peak less 2:", len(wave_peaks))
                    continue

                part_cards = img_math.seperate_card(gray_img, wave_peaks)#根据找出的波峰，分隔图片，从而得到逐个字符图片

                for i, part_card in enumerate(part_cards):
                    # 可能是固定车牌的铆钉

                    if np.mean(part_card) < 255 / 5:
                        print("a point")
                        continue
                    part_card_old = part_card

                    w = abs(part_card.shape[1] - SZ) // 2

                    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])#扩充图像边界
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    part_card = img_recognition.preprocess_hog([part_card])#提取特征向量
                    if i == 0:
                        resp = self.modelchinese.predict(part_card)
                        charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = self.model.predict(part_card)
                        charactor = chr(resp[0])
                    # 判断最后一个数是否是车牌边缘，假设车牌边缘被认为是1
                    if charactor == "1" and i == len(part_cards) - 1:
                        if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1太细，认为是边缘
                            continue
                    predict_result.append(charactor)

                roi = card_img
                card_color = color
                break
        return predict_result, roi, card_color  # 识别到的字符、定位的车牌图像、车牌颜色


    #文字检测方法
    def img_mser(self, filename):
        if type(filename) == type(""):
            img = img_math.img_read(filename)
        else:
            img = filename
        oldimg = img获取文本区域
        # 将不规则检测框处理成矩形框
        # 调用 MSER 算法
        mser = cv2.MSER_create(_min_area=600)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 灰度化
        regions, boxes = mser.detectRegions(gray)# 
        colors_img = []
        # 将不规则检测框处理成矩形框
        for box in boxes:
            x, y, w, h = box
            width, height = w, h
            if width < height:
                width, height = height, width
            ration = width / height

            if w * h > 1500 and 3 < ration < 4 and w > h:
                cropimg = img[y:y + h, x:x + w]
                colors_img.append(cropimg)

        debug.img_show(img)
        colors, car_imgs = img_math.img_color(colors_img)
        for i, color in enumerate(colors):
            if color != "no":
                print(color)
                debug.img_show(car_imgs[i])
