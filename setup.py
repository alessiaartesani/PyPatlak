from setuptools import setup, find_packages

setup(
    name='patlak_analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pydicom',
        'numpy',
        'nibabel',
        'scipy',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'patlak-analysis=patlak.patlak:main',
        ],
    },
    author='Alessia Artesani',
    description='A command-line tool for Patlak analysis of dynamic PET data.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/alessiaartesani/PyPatlak',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
)