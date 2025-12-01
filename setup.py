#!/usr/bin/env python
from setuptools import find_packages, setup
import re
import os


def readme():
    try:
        with open('README.md', encoding='utf-8') as f:
            return f.read()
    except IOError:
        return 'OpenSTL: Open-source Toolbox for SpatioTemporal Predictive Learning'


def get_version():
    """Reads the version from openstl/version.py."""
    version_file = 'openstl/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        # Use a simple regex to find the version string
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def parse_requirements(fname='requirements.txt', with_version=True):
   
    require_fpath = fname

    if not os.path.exists(require_fpath):
        return []

    requirements = []
    with open(require_fpath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('-r '):
                # Recursively parse requirements from another file
                requirements.extend(parse_requirements(line.split(' ')[1], with_version))
                continue

            if not with_version:
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                package_name = parts[0].strip()
                requirements.append(package_name)
            else:
                requirements.append(line)
    return requirements


if __name__ == '__main__':
    setup(
        name='OpenSTL',
        version=get_version(),
        description='OpenSTL: Open-source Toolbox for SpatioTemporal Predictive Learning',
        long_description=readme(),
        long_description_content_type='text/markdown',
        author='CAIRI Westlake University Contributors',
        author_email='lisiyuan@westlake.edu.com',
        keywords='spatiotemporal predictive learning, video prediction, '
        'unsupervised spatiotemporal learning',
        url='https://github.com/chengtan9907/OpenSTL',
        packages=find_packages(exclude=('configs', 'tools', 'demo', '*tests*', '*docs*')),
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12', 
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        license='Apache License 2.0',
        install_requires=parse_requirements('requirements/runtime.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'tests': parse_requirements('requirements/tests.txt'),
            'optional': parse_requirements('requirements/optional.txt'),
        },
        zip_safe=False)