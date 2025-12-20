# NST
This is the implementation of "Neural Style Transfer".<br>
Original paper: L. A. Gatys, A. S. Ecker, and M. Bethge. Image Style Transfer Using Convolutional Neural Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016. [link](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)

## Usage

### 0. Download pre-trained model
Please download VGG19 pre-trained model with ImageNet.
~~~
$ wget https://huggingface.co/koba-jon/pre-train_cpp/resolve/main/models/vgg19_bn.pth
~~~

### 1. Build
Please build the source file according to the procedure.
~~~
$ mkdir build
$ cd build
$ cmake ..
$ make -j4
$ cd ..
~~~

### 2. Dataset Setting

#### Setting

The following hierarchical relationships are recommended.

~~~
datasets
|--Dataset1
|    |--content.png
|    |--style.png
|
|--Dataset2
|--Dataset3
~~~

### 3. Image Generation

#### Example 1

- **Content: "Tuebingen Neckarfront" — Andreas Praefcke.** <br>
  Source: https://commons.wikimedia.org/wiki/File:Tuebingen_Neckarfront.jpg. <br>
  License: CC BY 3.0 Unported (https://creativecommons.org/licenses/by/3.0/). <br>
  Changes: resized for NST input.
- **Style: "The Starry Night" — Vincent van Gogh** <br>
  Source: https://commons.wikimedia.org/wiki/File:Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg <br>
  License: Public Domain <br>
  Changes: resized for NST input.

##### Setting
Please set the shell for executable file.
~~~
$ vi scripts/Neckarfront_TheStarryNight.sh
~~~
The following is an example of the generation phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='Neckarfront_TheStarryNight'

./NST \
    --generate true \
    --iterations 1000 \
    --dataset ${DATA} \
    --content "content.png" \
    --style "style.png" \
    --gpu_id 0
~~~

##### Run
Please execute the following to start the program.
~~~
$ sh scripts/Neckarfront_TheStarryNight.sh
~~~

#### Example 2

- **Content: "Tuebingen Neckarfront" — Andreas Praefcke.** <br>
  Source: https://commons.wikimedia.org/wiki/File:Tuebingen_Neckarfront.jpg. <br>
  License: CC BY 3.0 Unported (https://creativecommons.org/licenses/by/3.0/). <br>
  Changes: resized for NST input.
- **Style: "The Scream" — Edvard Munch** <br>
  Source: https://commons.wikimedia.org/wiki/File:The_Scream.jpg <br>
  License: Public Domain <br>
  Changes: resized for NST input.

##### Setting
Please set the shell for executable file.
~~~
$ vi scripts/Neckarfront_TheScream.sh
~~~
The following is an example of the generation phase.<br>
If you want to view specific examples of command line arguments, please view "src/main.cpp" or add "--help" to the argument.
~~~
#!/bin/bash

DATA='Neckarfront_TheScream'

./NST \
    --generate true \
    --iterations 1000 \
    --dataset ${DATA} \
    --content "content.png" \
    --style "style.png" \
    --gpu_id 0
~~~

##### Run
Please execute the following to start the program.
~~~
$ sh scripts/Neckarfront_TheScream.sh
~~~

