# 딥러닝 영상처리 실습 Image Transfer로 고흐 풍의 내 사진 만들기
## 안동대학교 컴퓨터공학과 20171132 
```
import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w  * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()


content_image_url = 'https://a.cdn-hotels.com/gdcs/production174/d1060/8ab26d42-60ff-42a4-913f-f0e925a58f18.jpg'  
style_image_url = 'https://www.artinsight.co.kr/data/tmp/1912/20191210235617_eostfpur.jpg'  

output_image_size = 384  # @param {type:"integer"}

content_img_size = (output_image_size, output_image_size)
style_img_size = (256, 256)  # Recommended to keep it at 256.

content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')
show_n([content_image, style_image], ['Content image', 'Style image'])

hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

outputs = hub_module(content_image, style_image)
stylized_image = outputs[0]
outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]
show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
```

# 소감
딥러닝을 활용하여 한개의 영상에다 하나의 이미지와 관련한 이미지로 변환하여 새로운 멋진 영상을 만들어내는 것이 너무 놀랍습니다.
여러가지 이미지를 가지고 원본이미지를 변환해보니 정말 재밌습니다. 
딥러닝은 우리들의 뇌처럼 구성된 알고리즘이며
심층학습을 거쳐 사람의 사고방식이 컴퓨터에게 가르치는 가계학습의 분야로
다양한 딥러닝 기법들이 컴퓨터 비전, 음성인식, 자연어 처리, 음성/신호처리 등의 분야에 적용되어 최첨단의 
결과들을 보여주듯이 컴퓨터가 학습한다는 것에 너무 놀랍고 기술이 엄청 빨리 상상이상으로 바뀌고 있는 거 같습니다
인터넷에서 검색하면서 알게 된 것이 머신러닝과 딥러닝이 있었는데
머신러닝은 기존의 머신러닝에서 학습하려는 데이터의 여러 특징중에서 어떤 특징을 추출할지
사람이 직접 분석하고 판단해야만 했다면
딥러닝은  기계가 자동으로 학습하려는 데이터 특징을 추출하여 학습까지 하게 됩니다
그렇게 딥러닝은 기계의 자가 학습여부로 머신러닝과 차이점이 있으며
딥러닝은 대규모 데이터에서 중요한 패턴 및 규칙을 학습하고 ,
이를 토대로 의사결정이나 예측을 수행할 수 있습니다
그리고 딥러닝에서 가장 기본이 되는것은 인공신경망으로
딥러닝은  인간의 뇌와 같이 수많은 뉴런들이 신호를 전달하듯이
컴퓨터가 이러한 신호를 기반으로 다양한 사고를 구현하도록 노력한 것이 딥러닝의 인공신경망입니다


