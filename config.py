import pathlib

_root = pathlib.Path(__file__).parent


def get_save_path(kind):
    return str(pathlib.Path.joinpath(_root, 'checkpoints', '{}.pkl'.format(kind)))
