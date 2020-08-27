# Introduction
This project aims to perform Cat Prey Detection with Deep Learning on any cat in any environement. For a brief and light intro in what it does, check out the [Raspberry Pi blog post](https://www.raspberrypi.org/blog/deep-learning-cat-prey-detector/)  about it. This 
project utilises a cascade of Convolutional Neural Networks (CNN) to process images and infer about the Prey/No_Prey state of a cat image. The reason why it uses a cascade is simple: CNN's need a huge amount of data to learn their task, the amount is related to the complexity of the problem. For general cat prey detection, a NN would need to first learn what a cat is in general, and find out how their snouts differ with and without prey. This turns out to be quite complex for a machine to learn and we simply don't have enough data of cats with prey (only 150 images to be exact). This is why we use a cascade to break up the complex problem into smaller substages

- First detect if there is a cat or not. There exists a lot of data on this problem and a lot of complete solutions such for example any COCO trained Object detector such as for example [YOLOV3 on COCO](https://pjreddie.com/darknet/yolo/) or [Tensorflows COCO trained MobileNetV2](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). 

- Second detect the snout of the cat within the image section of the first stage.


Here is a brief overview of the cascade:

<img src="/readme_images/cascade.png" width="400">


The Catfinder Stage utilises Tensorflows object detection API and runs a pretrained (by Tensorflow) COCO MobileNetV2 architecture which aims to detect a cat in the image. The next stage aims to detect the Snout of the cat, it is a combination of different Computer Vision (CV) techniques such as HAAR-Cascade and a CNN (CNN-BB + FF).
