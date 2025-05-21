import tensorflow as tf
import numpy as np
from pathlib import Path
import os, cv2, time, sys

from object_detection.utils import label_map_util

cat_cam_py = str(Path(os.getcwd()).parents[0])
#print('CatCamPy: ', cat_cam_py)
PC_models_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/models/Prey_Classifier')
FF_models_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/models/Face_Fur_Classifier')
EYE_models_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/models/Eye_Detector')
HAAR_models_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/models/Haar_Classifier')
CR_models_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/models/Cat_Recognizer')


class CC_MobileNet_Stage():
    def __init__(self):
        self.MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
        self.img_org = None

        # Start the CNN
        self.sess, self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections, self.image_tensor, self.category_index = self.init_cnn_model()

    def init_cnn_model(self):
        #### Initialize TensorFlow model ####

        # This is needed since the working directory is the object_detection folder.
        sys.path.append('..')

        # Grab path to current working directory
        #print(os.environ['PYTHONPATH'].split(os.pathsep)[1])
        TF_OD_PATH = os.environ['PYTHONPATH'].split(os.pathsep)[1] + '/object_detection'
        #print(TF_OD_PATH)

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(TF_OD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(TF_OD_PATH, 'data', 'mscoco_label_map.pbtxt')

        # Number of classes the object detector can identify
        NUM_CLASSES = 90

        ## Load the label map.
        # Label maps map indices to category names, so that when the convolution
        # network predicts `5`, we know that this corresponds to `airplane`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # print(categories)
        # print(category_index)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.compat.v1.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        #print('CNN is ready to go!')

        return sess, detection_boxes, detection_scores, detection_classes, num_detections, image_tensor, category_index

    def resize_img(self, input_img):
        # prepare input image to shape (300, 300, 3) as the MobileNetV2 model specifies
        self.img_org = input_img
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        return img

    def do_cc(self, target_img):
        preprocessed_img = self.resize_img(input_img=target_img)
        pred_class, pred_cc_bb, inference_time = self.pet_detector(preprocessed_img, self.sess, self.detection_boxes,
                                                                    self.detection_scores, self.detection_classes,
                                                                    self.num_detections, self.image_tensor,
                                                                    self.category_index)
        #print('CC_time: %s', inference_time)
        return pred_cc_bb, pred_class, inference_time

    def draw_rectangle(self, img, box, color, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 3
        text_pos = (box[0][0], int(box[0][1]-16))

        cv2.putText(img, text,
                    text_pos,
                    font,
                    fontScale,
                    color,
                    lineType)
        return cv2.rectangle(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, 5)


    # This function contains the code to detect a pet, determine if it's
    # inside or outside, and send a text to the user's phone.
    def pet_detector(self, frame, sess, detection_boxes, detection_scores, detection_classes, num_detections, image_tensor, category_index):
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        start_time = time.time()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        end_time = time.time()
        inference_time = end_time - start_time

        # Check the class of the top detected object by looking at classes[0][0].
        # If the top detected object is a cat (17) or a dog (18) (or a teddy bear (88) for test purposes),
        # find its center coordinates by looking at the boxes[0][0] variable.
        # boxes[0][0] variable holds coordinates of detected objects as (ymin, xmin, ymax, xmax)
        xmin = int(boxes[0][0][1] * self.img_org.shape[1])
        ymin = int(boxes[0][0][0] * self.img_org.shape[0])
        xmax = int(boxes[0][0][3] * self.img_org.shape[1])
        ymax = int(boxes[0][0][2] * self.img_org.shape[0])
        target_box = np.array([(xmin,ymin), (xmax,ymax)]).reshape((-1, 2))



        if (int(classes[0][0]) == 17 or int(classes[0][0]) == 18):
            return True, target_box, inference_time

        else:
            return False, target_box, inference_time

class Haar_Stage():
    def __init__(self):
        self.TARGET_SIZE = 224
        self.inference_time_list = []

        self.bb_model_name = ''
        self.IoU = 0
        self.ACC = 0

        self.models_dir = HAAR_models_dir
        self.model = 'haarcascade_frontalcatface_extended.xml'
        self.face_cascade = cv2.CascadeClassifier(os.path.join(HAAR_models_dir, self.model))

    def haar_do(self, target_img, full_img, cc_bbs):
        pred_bb, inference_time, haar_found_bool = self.haar_predict(input=target_img)
        #print('Haar_time: ', str('%.2f' % inference_time))

        pred_bb_full = pred_bb[:]

        pred_bb_full[0][0] = max(pred_bb[0][0] + cc_bbs[0][0], 0)
        pred_bb_full[0][1] = max(pred_bb[0][1] + cc_bbs[0][1], 0)
        pred_bb_full[1][0] = min(pred_bb[1][0] + cc_bbs[0][0], full_img.shape[1])
        pred_bb_full[1][1] = min(pred_bb[1][1] + cc_bbs[0][1], full_img.shape[0])

        return pred_bb_full, inference_time, haar_found_bool

    def haar_predict(self, input):
        start_time = time.time()
        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1, minSize=(25, 25))
        inference_time = time.time() - start_time

        haar_found_bool = False
        if len(faces) != 0:
            face = max(faces, key=lambda coords: abs(coords[0] - coords[2]) * abs(coords[1] - coords[3])).reshape((-1, 2))
            pred_bb = face[:]
            pred_bb[0][0] = int(face[0][0] - face[1][0] * 0.2)
            pred_bb[0][1] = int(face[0][1] - face[1][1] * 0.4)
            pred_bb[1][0] = int(face[0][0] + face[1][0] * 1.2)
            pred_bb[1][1] = int(face[0][1] + face[1][1] * 1.6)
            haar_found_bool = True

        else:
            pred_bb = np.array([(0,0), (0,0)]).reshape((-1, 2))


        return pred_bb, inference_time, haar_found_bool

    def draw_rectangle(self, img, box, color, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 3
        text_pos = (box[0][0], int(box[0][1]-16))

        cv2.putText(img, text,
                    text_pos,
                    font,
                    fontScale,
                    color,
                    lineType)
        return cv2.rectangle(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, 5)


    def calc_iou(self, gt_bbox, pred_bbox):
        (x_topleft_gt, y_topleft_gt), (x_bottomright_gt, y_bottomright_gt) = gt_bbox.tolist()
        (x_topleft_p, y_topleft_p), (x_bottomright_p, y_bottomright_p) = pred_bbox.tolist()

        if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
            raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)

        # if the GT bbox and predcited BBox do not overlap then iou=0
        if (x_bottomright_gt < x_topleft_p):# If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
            return 0.0
        if (y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
            return 0.0
        if (x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
            return 0.0
        if (y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
            return 0.0

        GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
        Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

        x_top_left = np.max([x_topleft_gt, x_topleft_p])
        y_top_left = np.max([y_topleft_gt, y_topleft_p])
        x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
        y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

        intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

        union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

        return intersection_area / union_area

class PC_Stage():
    def __init__(self):
        self.TARGET_SIZE = 224
        self.fp = 0
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.inference_time_list = []

        #Handle args
        self.models_dir = PC_models_dir
        self.pc_model_name = '0.86_512_05_VGG16_ownData_FTfrom15_350_Epochs_2020_05_15_11_40_56.h5'

        dependencies = {
            'get_f1': self.get_f1
        }
        if 'F1' in self.pc_model_name:
            self.pc_model = tf.keras.models.load_model(os.path.join(PC_models_dir, self.pc_model_name),
                                                       custom_objects=dependencies, compile=False)
        else:
            self.pc_model = tf.keras.models.load_model(os.path.join(PC_models_dir, self.pc_model_name), compile=False)

    def get_f1(self, y_true, y_pred):  # taken from old keras source code
        K = tf.keras.backend
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    def resize_img(self, img_org):
        return cv2.resize(img_org, (self.TARGET_SIZE, self.TARGET_SIZE)) * (1. / 255)

    def pc_prediction(self, img, pc_model):
        preprocessed_img = self.resize_img(img_org=img).reshape((1, self.TARGET_SIZE, self.TARGET_SIZE, 3))
        start_time = time.time()
        class_pred = pc_model.predict(preprocessed_img)
        inference_time = time.time() - start_time

        return class_pred[0][0], inference_time


    def pc_do(self, target_img):
        pred_val, inference_time = self.pc_prediction(img=target_img, pc_model=self.pc_model)

        if pred_val <= 0.5:
            return False, pred_val, inference_time
        else:
            return True, pred_val, inference_time

    def input_text(self, img, text, text_pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 3

        cv2.putText(img, text,
                    text_pos,
                    font,
                    fontScale,
                    color,
                    lineType)
        return img

class FF_Stage():
    def __init__(self):
        self.TARGET_SIZE = 224
        self.fp = 0
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.inference_time_list = []

        # Handle args
        self.models_dir = FF_models_dir
        self.ff_model_name = '256_05_mobileNet_50_Epochs_2020_05_07_14_56_25.h5'
        self.ff_model = tf.keras.models.load_model(os.path.join(self.models_dir, self.ff_model_name), compile=False)

    def resize_img(self, img_org):
        return cv2.resize(img_org, (self.TARGET_SIZE, self.TARGET_SIZE)) * (1. / 255)


    def ff_prediction(self, img, ff_model):
        preprocessed_img = self.resize_img(img_org=img).reshape((1, self.TARGET_SIZE, self.TARGET_SIZE, 3))

        start_time = time.time()
        class_pred = ff_model.predict(preprocessed_img)
        inference_time = time.time() - start_time

        return class_pred[0][0], inference_time


    def ff_do(self, target_img):
        pred, inference_time = self.ff_prediction(img=target_img, ff_model=self.ff_model)

        if pred <= 0.5:
            return True, pred, inference_time
        else:
            return False, pred,  inference_time

class Eye_Stage():
    def __init__(self):
        self.TARGET_SIZE = 224
        model_name = 'trainwhole100_Epochs_2020_04_30_18_05_25.h5'
        self.eye_model = tf.keras.models.load_model(os.path.join(EYE_models_dir, model_name), compile=False)

    def resize_img(self, img_resize):
        old_size = img_resize.shape[:2]  # old_size is in (height, width) format
        ratio = float(self.TARGET_SIZE) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        # new_size should be in (width, height) format
        img_resize = cv2.resize(img_resize, (new_size[1], new_size[0]))
        delta_w = self.TARGET_SIZE - new_size[1]
        delta_h = self.TARGET_SIZE - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        img_resize = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img_resize, top, left

    def eye_prediction(self, input):
        inputs = (input.astype('float32') / 255).reshape((1, self.TARGET_SIZE, self.TARGET_SIZE, 3))
        pred_eyes = self.eye_model.predict(inputs)[0].reshape((-1, 2))
        return pred_eyes

    def eye_full_prediction(self, target_img, cc_bbs):
        preprocessed_img, top, left = self.resize_img(target_img)
        inputs = (preprocessed_img.astype('float32') / 255).reshape((1, self.TARGET_SIZE, self.TARGET_SIZE, 3))
        start_time = time.time()
        pred_eyes = self.eye_model.predict(inputs)[0].reshape((-1, 2))
        inference_time = time.time() - start_time

        ratio_h = self.TARGET_SIZE / target_img.shape[0]
        ratio_w = self.TARGET_SIZE / target_img.shape[1]

        pred_eyes[0][0] = int(pred_eyes[0][0] / ratio_w) + cc_bbs[0][0] - left
        pred_eyes[0][1] = int(pred_eyes[0][1] / ratio_h) + cc_bbs[0][1] - top
        pred_eyes[1][0] = int(pred_eyes[1][0] / ratio_w) + cc_bbs[0][0] - left
        pred_eyes[1][1] = int(pred_eyes[1][1] / ratio_h) + cc_bbs[0][1] - top

        return pred_eyes, inference_time

    def eyes_to_box(self, pred_eyes, cc_target_img, cc_pred_full):
        x_mid = int((pred_eyes[0][0] + pred_eyes[1][0]) / 2)
        y_mid = int((pred_eyes[0][1] + pred_eyes[1][1]) / 2)
        x_diff = abs(pred_eyes[0][0] - pred_eyes[1][0])
        y_diff = abs(pred_eyes[0][1] - pred_eyes[1][1])
        xy_diag = (x_diff**2 + y_diff**2)**0.5


        cc_bb_x_diff = abs(cc_pred_full[0][0] - cc_pred_full[1][0])
        cc_bb_y_diff = abs(cc_pred_full[0][1] - cc_pred_full[1][1])
        cc_bb_diff = max(cc_bb_x_diff, cc_bb_y_diff)

        top_margin = cc_bb_diff / 4
        bottom_margin = cc_bb_diff / 3
        side_margin = cc_bb_diff / 4

        xmin = max(int(x_mid - side_margin), 0)
        ymin = max(int(y_mid - top_margin), 0)
        xmax = min(int(x_mid + side_margin), cc_target_img.shape[1])
        ymax = min(int(y_mid + bottom_margin), cc_target_img.shape[0])

        bbs = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
        return bbs

    def do_eyes(self, cc_target_img, eye_target_img, cc_pred_bb):
        eyes_full_coords, inference_time = self.eye_full_prediction(target_img=eye_target_img, cc_bbs=cc_pred_bb)
        bbs = self.eyes_to_box(pred_eyes=eyes_full_coords, cc_target_img=cc_target_img, cc_pred_full=cc_pred_bb)

        #for eye in eyes_full_coords:
        #    cv2.circle(cc_target_img, center=tuple(eye), radius=15, color=(0, 255, 0), thickness=5)

        pc_xmin = int(bbs[0][0])
        pc_ymin = int(bbs[0][1])
        pc_xmax = int(bbs[1][0])
        pc_ymax = int(bbs[1][1])
        snout_crop = cc_target_img[pc_ymin:pc_ymax, pc_xmin:pc_xmax].copy()

        return snout_crop, bbs, inference_time
