import click
from models import Net


@click.command()
@click.option('--size', default=32, type=int, help='Number of batch size.')
@click.option('--epoch', default=10, type=int, help='Number of epoch.')
@click.option('--kind', prompt=True, type=click.Choice(['style', 'color', 'reflex', 'space', 'thy']),
              help='kind of model.')
@click.option('--mode', prompt=True, type=click.Choice(['train', 'test']), help='mode of model.')
def main(size, epoch, kind, mode):
    net = Net(kind, mode, size, epoch)
    if mode == 'train':
        net.get_params_number()
        net.train()
    else:
        net.test()


if __name__ == '__main__':
    main()
