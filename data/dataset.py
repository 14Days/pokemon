import pathlib
from torch.utils.data import Dataset
from PIL import Image
from data.get_data import load_image
from data.transforms import _image_transforms


class MyDataset(Dataset):
    def __init__(self, kind: str, mode: str, net: str):
        self.transform = f'{net}_{mode}'
        # 加载图片路径
        path = pathlib.Path(__file__).parent
        path = pathlib.Path.joinpath(path, 'set', kind)
        self.images, self.labels, self.name2label = load_image(str(path), mode)

    def __getitem__(self, index):
        image_name = self.images[index]
        image = Image.open(image_name).convert("RGB")
        image_as_tensor = _image_transforms[self.transform](image)
        label = self.labels[index]
        return image_as_tensor, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    temp = MyDataset('color', 'train', 'BnOpt')
    print(temp.__getitem__(5))
