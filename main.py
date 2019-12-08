import click


@click.command()
@click.option('--size', default=32, type=int, help='Number of batch size.')
@click.option('--epoch', default=10, type=int, help='Number of epoch.')
@click.option('--kind', prompt=True, type=click.Choice(['style', 'color', 'reflex', 'space']), help='kind of model.')
@click.option('--mode', prompt=True, type=click.Choice(['train', 'test']), help='mode of model.')
def main(size, epoch, kind, mode):
    print(size, epoch, kind, mode)


if __name__ == '__main__':
    main()
