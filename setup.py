from setuptools import setup

setup(
    name='semseg',
    version='0.1.0',
    author='Bashir Kazimi',
    author_email='kazimibashir907@gmail.com',
    packages=['semseg', 'semseg.test'],
    url='http://pypi.python.org/pypi/al/',
    license='LICENSE.txt',
    description='A Pytorch Semantic Segmentation Framework',
    long_description=open('README.txt').read(),
    install_requires=[
    ],
)
