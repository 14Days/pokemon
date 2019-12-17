import pathlib

_root = pathlib.Path(__file__).parent


def get_save_path(kind, model):
    return str(pathlib.Path.joinpath(_root, 'checkpoints', f'{kind}_{model}.pkl'))


if __name__ == '__main__':
    print(get_save_path('color', 'VGG16'))
