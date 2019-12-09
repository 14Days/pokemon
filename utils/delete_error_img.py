import pathlib
import csv
import os
import click
from PIL import Image


@click.command()
@click.option('--dir', prompt=True, type=click.Choice(['style', 'color', 'reflex', 'space']), help='kind of model.')
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
    delete_image()
