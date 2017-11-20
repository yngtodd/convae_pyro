try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name = 'convae_pyro',
    packages = ['convae_pyro'],
    install_requires=['numpy', 'pytorch', 'pyro', 'matplotlib', 'seaborn', 'scipy',
                      'sklearn', 'pandas'],
    version = '0.0.1',
    description = 'A convolutional VAE learned through variational inference',
    author = 'Todd Young',
    author_email = 'young.todd.mk@gmail.com',
    url = 'https://github.com/yngtodd/convae_pyro',
    keywords = ['variational inference', 'convolution', 'variational autoencoder'],
)
