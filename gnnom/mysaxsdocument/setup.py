from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='saxsdocument',
      version='1.1',
      description='Library for readin/writing SAXS related files',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Development status :: DEBUG',
          'License :: MIT',
          'Programming language :: Python 3',
          'Operating System :: OS Independent'
      ],
      url='https://github.com/DimaMolod/saxsdocument',
      author='Dima',
      author_email='dmolodenskiy@embl-hamburg.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[],
      include_package_data=True,
      zip_safe=False)
