import click
from models import Net


@click.command()
@click.option('--size', default=32, type=int, help='Number of batch size.')
@click.option('--epoch', default=10, type=int, help='Number of epoch.')
@click.option('--lr', default=0.0001, type=int, help='Number of learn rate')
@click.option('--name', prompt=True, type=str, help='Name of model')
@click.option('--kind', prompt=True, type=click.Choice(['style', 'color', 'reflex', 'space', 'thy']),
              help='Kind of dataset.')
@click.option('--mode', prompt=True, type=click.Choice(['train', 'test']), help='Mode of model.')
def main(size, epoch, lr, name, kind, mode):
    net = Net(name, kind, mode, size, epoch, lr)
    if mode == 'train':
        net.get_params_number()
        net.train()
    else:
        net.test()


if __name__ == '__main__':
    main()
