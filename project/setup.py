from setuptools import setup
import os, sys

sys.path.append(os.path.dirname(__file__))

setup(
    name='yolodetector',
    version=0.1,
    description='Package of YOLO detection',
    author='',
    author_email='ziyi.zhao@epfl.ch',
    packages=[
        'yolodetector',
        'yolodetector.layers',
    ],
)
