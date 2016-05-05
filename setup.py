#!/usr/bin/env python

from distutils.command.build_py import build_py as BuildPyCommand
import os
import os.path as osp
import shlex
import subprocess
import sys
import tempfile

from setuptools import find_packages
from setuptools import setup


version = '1.3.5'


if sys.argv[-1] == 'release':
    commands = [
        'python setup.py sdist',
        'twine upload dist/fcn-{0}.tar.gz'.format(version),
        'git tag v{0}'.format(version),
        'git push origin master --tag',
    ]
    for cmd in commands:
        subprocess.call(shlex.split(cmd))
    sys.exit(0)


class FcnBuildPyCommand(BuildPyCommand):
    def run(self):
        BuildPyCommand.run(self)
        # create data dirs
        data_dir = osp.join(self.build_lib, 'fcn/_data')
        if not osp.exists(data_dir):
            os.makedirs(data_dir)
        # download chainermodel to the data dir
        output = osp.join(data_dir, 'fcn8s.chainermodel')
        url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2veTdBQWZybENLWmM'
        cmd = "gdown -q '{0}' -O {1}".format(url, output)
        print("Downloading '{0}' by command '{1}'".format(output, cmd))
        subprocess.check_call(shlex.split(cmd))
        BuildPyCommand.run(self)


setup(
    name='fcn',
    version=version,
    packages=find_packages(),
    cmdclass={'build_py': FcnBuildPyCommand},
    scripts=['scripts/fcn_forward.py'],
    install_requires=open('requirements.txt').readlines(),
    description='Fully Convolutional Networks',
    long_description=open('README.rst').read(),
    author='Kentaro Wada',
    author_email='www.kentaro.wada@gmail.com',
    url='http://github.com/wkentaro/fcn',
    license='MIT',
    keywords='machine-learning',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Topic :: Internet :: WWW/HTTP',
    ],
)
