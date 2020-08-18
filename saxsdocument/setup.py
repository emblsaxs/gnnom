from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='saxsdocumnet',
      version = '0.0.1',
      description = 'Library for dealing with SAXS related files',
      long_description = readme(),
      long_description_content_type = 'text/markdown',
      classifiers = [
          'Development status :: DEBUG',
          'License :: MIT',
          'Programming language :: Python 3',
          'Operating System :: OS Independent'
      ],
      url = 'https://github.com/DimaMolod/saxsdocument',
      author = 'DimaMolodenskiy',
      author_email = 'dmolodenskiy@embl-hamburg.de',
      keywords = 'core package',
      license = 'MIT',
      packages = ['saxsdocument'],
      install_requires = [],
      include_package_data=True,
      zip_safe = False)