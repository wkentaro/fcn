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


version = '1.2.2'


if sys.argv[-1] == 'release':
    commands = [
        'python setup.py sdist upload',
        'git tag v{0}'.format(version),
        'git push origin master --tag',
    ]
    for cmd in commands:
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


class FcnBuildPyCommand(BuildPyCommand):
    def run(self):
        output_dir = osp.join(self.build_lib, 'fcn/_data')
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        output = osp.join(output_dir, 'fcn8s.chainermodel')
        url = 'https://drive.google.com/uc?id=0B9P1L--7Wd2veTdBQWZybENLWmM'
        print("Downloading '{0}' from '{1}'".format(output, url))
        subprocess.check_call(['gdown', '-q', url, '-O', output])
        BuildPyCommand.run(self)


setup(
    name='fcn',
    version=version,
    packages=find_packages(),
    cmdclass={'build_py': FcnBuildPyCommand},
    scripts=['scripts/fcn_forward.py'],
    install_requires=open('requirements.txt').readlines(),
    setup_requires=['gdown'],
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
