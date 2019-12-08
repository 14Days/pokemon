import os
import glob
import random
import csv


def load_image(root, mode='train'):
    """
    加载类别
    :param root: 训练素材的根文件夹
    :param mode: 加载模式
    :return:
    """
    # 创建编码表
    name2label = {}
    # 遍历文件夹得到分类，注意进行排序
    for name in sorted(os.listdir(os.path.join(root))):
        if os.path.isdir(os.path.join(root, name)):
            name2label[name] = len(name2label.keys())

    images, labels = _load_csv(root, 'image.csv', name2label)
    if mode == 'train':
        images = images[:int(0.8 * len(images))]
        labels = labels[:int(0.8 * len(labels))]
    else:
        images = images[int(0.8 * len(images)):]
        labels = labels[int(0.8 * len(labels)):]

    return images, labels, name2label


def _load_csv(root, filename, name2label: dict) -> (list, list):
    """
    加载图片路径与类型
    :param root: 根文件夹路径
    :param filename: csv文件名
    :param name2label: 类别对象
    :return:
    """
    if not os.path.exists(os.path.join(root, filename)):
        images = []
        for name in name2label.keys():
            images += glob.glob(os.path.join(root, name, '*.png'))
            images += glob.glob(os.path.join(root, name, '*.jpg'))
            images += glob.glob(os.path.join(root, name, '*.jpeg'))
        # 打散图片
        random.shuffle(images)
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            write = csv.writer(f)
            for img in images:
                name = img.split(os.sep)[-2]
                label = name2label[name]
                write.writerow([img, label])

    images, labels = [], []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            label = int(label)
            images.append(img)
            labels.append(label)

    return images, labels
