import click


@click.command()
@click.option('--size', default=32, type=int, help='Number of batch size.')
@click.option('--epoch', default=10, type=int, help='Number of epoch.')
@click.option('--kind', prompt=True, type=click.Choice(['style', 'color', 'reflex', 'space']), help='kind of model.')
def main(size, epoch, kind):
    print(size, epoch, kind)


if __name__ == '__main__':
    main()
