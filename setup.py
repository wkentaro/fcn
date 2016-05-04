#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup


version = '1.1.0'


setup(
    name='fcn',
    version=version,
    packages=find_packages(),
    package_data={'fcn': ['_data/fcn8s.chainermodel']},
    scripts=['scripts/fcn_forward.py'],
    install_requires=open('requirements.txt').readlines(),
    description='Fully Convolutional Networks',
    long_description=open('README.rst').read(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='http://github.com/wkentaro/fcn',
    license='MIT',
    keywords='Machine Learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Topic :: Internet :: WWW/HTTP',
    ],
)
