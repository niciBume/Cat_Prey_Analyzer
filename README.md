# Introduction
If you own a cat that has the freedom to go outside, then you probably are familliar with the issue of your feline bringing home prey. This leads to a clean up effort that one wants to avoid!
This project aims to perform Cat Prey Detection with Deep Learning on any cat in any environement. For a brief and light intro in what it does, check out the [Raspberry Pi blog post](https://www.raspberrypi.org/blog/deep-learning-cat-prey-detector/) about it. The idea is that you can use the output of this system to trigger your catflap such that it locks out you cat.

# Related work
This isn't the first approach at solving the mentioned problem! There have been other equally (if not better) valid approaches such as the [Catcierge](https://github.com/JoakimSoderberg/catcierge) which analyzes the silhouette of the cat a very recent approach of the [AI powered Catflap](https://www.theverge.com/tldr/2019/6/30/19102430/amazon-engineer-ai-powered-catflap-prey-ben-hamm).
The difference of this project however is that it aims to solve *general* cat-prey detection through a vision based approach. Meaning that this should work for any cat! 


# Architecture
This project utilises a cascade of Convolutional Neural Networks (CNN) to process images and infer about the Prey/No_Prey state of a cat image. The reason why it uses a cascade is simple: CNN's need data to learn their task, the amount of data is related to the complexity of the problem. For general cat prey detection, a NN would need to first learn what a cat is in general, and find out how their snouts differ with and without prey. This turns out to be quite complex for a machine to learn and we simply don't have enough data of cats with prey (only 150 images to be exact). This is why we use a cascade to break up the complex problem into smaller substages:

- First detect if there is a cat or not. There exists a lot of data on this problem and a lot of complete solutions such for example any COCO trained Object detector such as for example [Tensorflows COCO trained MobileNetV2](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). We call it the CatFinder stage which utilises the mentioned Tensorflow object detection API and runs the Tensorflow pretrained MobileNetV2 and soley aims to detect a cat in the image.

- Second we detect the snout of the cat within the image section of the first stage. This is done combination of different Computer Vision (CV) techniques such as HAAR-Cascade and a self trained CNN (CNN-BB + FF).

- Lastly we classify the snout-cropped image of the cat with a self trained CNN based on the VGG16 architecture. It was only trained with ~150 Prey-Snout-Images gathered from the internet and personal images. This is the data-critical section; we can find more than enough images of cats but only very few images of cats with prey in their snout. Obviously the tasks complexity of identifying prey in a cropped image of the snout is simpler than classifying so on a full image, hence the extra steps of the cascade.

Here is a brief overview of the cascade:

<img src="/readme_images/cascade.png" width="400">

As depicted in the image, there are four resulting paths that can be taken which yield different runtimes. On an off the shelf Raspberry Pi 4 the runtimes areas follows:

- P_1 507 ms
- P_2 3743 ms
- P_3 2035 ms
- P_4 5481 ms

Now the runtime numbers are quite high, which is why we use a dynamically adapting queue to adjust the framerate of the system. This part is built specifically for the RPI and its camera system. It is a multithreading process where the camera runs on an own thread and the cascade on a seperate thread. The camera fills a concurrent queue while the cascade pops the queue at a dynamic rate. Sounds fancy and complicated, but it isn't:

<img src="/readme_images/queue.png" width="400">

# Results

<img src="/readme_images/combined_curve.png" width="700">


<img src="/readme_images/Cummuli Confusion Matrix @ Threshold_ 2.96.png" width="400">


