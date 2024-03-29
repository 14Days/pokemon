from torch.utils.data import DataLoader
from data.dataset import MyDataset


def get_data_loader(kind, mode, batch_size, net):
    data_set = MyDataset(kind, mode, net)
    return DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True), data_set
