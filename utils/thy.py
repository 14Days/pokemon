import os
import shutil
import pathlib
from xml.dom.minidom import parse


def create():
    root = pathlib.Path(__file__).parent.parent
    root = pathlib.Path.joinpath(root, 'data', 'set', 'thy')
    annotation = pathlib.Path.joinpath(root, 'Annotations')
    image = pathlib.Path.joinpath(root, 'images')
    label_set = set()
    for item in list(os.listdir(str(annotation))):
        xml_path = pathlib.Path.joinpath(annotation, item)
        dom_obj = parse(str(xml_path)).documentElement
        label = dom_obj.getElementsByTagName('object')[0].getElementsByTagName('name')[0].firstChild.data
        label_set.add(label)

    for item in label_set:
        if not os.path.exists(str(pathlib.Path.joinpath(root, item))):
            os.mkdir(str(pathlib.Path.joinpath(root, item)))

    for item in list(os.listdir(str(annotation))):
        xml_path = pathlib.Path.joinpath(annotation, item)
        dom_obj = parse(str(xml_path)).documentElement
        file_name = dom_obj.getElementsByTagName('filename')[0].firstChild.data
        label = dom_obj.getElementsByTagName('object')[0].getElementsByTagName('name')[0].firstChild.data
        file_name = pathlib.Path.joinpath(image, file_name)
        try:
            shutil.copy(str(file_name), str(pathlib.Path.joinpath(root, label)))
        except:
            continue


if __name__ == '__main__':
    create()
