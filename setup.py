#!/usr/bin/env python

import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '6.4.7'


if sys.argv[-1] == 'release':
    commands = [
        'git tag v{}'.format(version),
        'git push origin master --tag',
        'python setup.py sdist upload',
    ]
    for cmd in commands:
        subprocess.call(shlex.split(cmd))
    sys.exit(0)


setup(
    name='fcn',
    version=version,
    packages=find_packages(),
    include_package_data=True,
    scripts=['scripts/fcn_infer.py'],
    install_requires=open('requirements.txt').readlines(),
    description='Fully Convolutional Networks',
    long_description=open('README.md').read(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='http://github.com/wkentaro/fcn',
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
