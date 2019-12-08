import pathlib
import csv
import os
from PIL import Image


def delete_image(dir):
    path = pathlib.Path(__file__).parent.parent
    path = pathlib.Path.joinpath(path, 'data', 'set', dir, 'image.csv')
    with open(str(path)) as f:
        reader = csv.reader(f)
        for row in reader:
            img = row[0]
            print(img)
            try:
                Image.open(img).convert('RGB')
            except OSError:
                print('delete')
                os.remove(img)
                continue



if __name__ == '__main__':
    delete_image('color')
