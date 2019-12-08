import pathlib

_root = pathlib.Path(__file__).parent


def get_save_path(kind):
    return str(pathlib.Path.joinpath(_root, 'checkpoints', f'{kind}.pkl'))


if __name__ == '__main__':
    print(get_save_path('color'))
