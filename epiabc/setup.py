from setuptools import setup

setup(name='epiabc',
      version='0.1',
      description='the epiabc algorithm',
      url='https://github.com/jooooooy/sta663',
      author='ajjy',
      author_email='jingyi.linxmu@gmail.com',
      license='STA663',
      install_requires=['numpy','scipy', 'numba'],
      packages=['epiabc'],
      package_data  = {
        "epiabc": ["*.dat"], # or "canbeAny": ["dataset/*.data"]
    },
      zip_safe=False)
