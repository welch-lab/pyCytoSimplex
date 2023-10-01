from setuptools import setup, find_packages

setup(
    name='CytoSimplex',
    version='0.1.0',
    author=['Yichen Wang', 'Jialin Liu'],
    author_email='wayichen@umich.edu',
    description="""CytoSimplex is module that creates simplex plot showing
                   similarity between single-cells and clusters, while being
                   able to add velocity as another layer of information""",
    # long_description='A longer description of your project',
    long_description_content_type='text/markdown',
    url='https://github.com/mvfki/pyCytoSimplex',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'sklearn',
        'numpy',
        'anndata',
        'scipy',
        'pandas',
        'matplotlib',
        'mpltern'
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    keywords='python package distribution',
    project_urls={
        'Source': 'https://github.com/mvfki/pyCytoSimplex',
    },
)
