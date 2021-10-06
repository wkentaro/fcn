from __future__ import print_function

import distutils.spawn
import shlex
import subprocess
import sys

from setuptools import find_packages
from setuptools import setup


version = '6.4.20'


if sys.argv[1] == 'release':
    if not distutils.spawn.find_executable('twine'):
        print(
            'Please install twine:\n\n\tpip install twine\n',
            file=sys.stderr,
        )
        sys.exit(1)

    commands = [
        'git pull origin main',
        'git tag v{:s}'.format(version),
        'git push origin main --tags',
        'python setup.py sdist',
        'twine upload dist/fcn-{:s}.tar.gz'.format(version),
    ]
    for cmd in commands:
        print('+ {}'.format(cmd))
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


def get_long_description():
    with open('README.md') as f:
        long_description = f.read()

    try:
        import github2pypi

        return github2pypi.replace_url(
            slug='wkentaro/fcn', content=long_description
        )
    except Exception:
        return long_description


def get_install_requires():
    with open('requirements.txt') as f:
        return [req.strip() for req in f]


setup(
    name='fcn',
    version=version,
    packages=find_packages(exclude=['github2pypi']),
    include_package_data=True,
    scripts=['scripts/fcn_infer.py'],
    install_requires=get_install_requires(),
    description='Fully Convolutional Networks',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
    ],
)
