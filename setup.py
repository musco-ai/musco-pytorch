import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


class InstallLocalPackage(install):
    def run(self):
        install.run(self)
        subprocess.call("cd dependencies/tensorly && python setup.py install --force&& cd ../..", shell=True)


try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements


def load_requirements(file_name):
    requirements = parse_requirements(file_name, session="test")
    return [str(item.req) for item in requirements]


setup(
    name="musco-pytorch",
    version="v1.0.1",
    description="MUSCO: Multi-Stage COmpression of Neural Networks",
    author="Julia Gusak, Maksym Kholiavchenko, Evgeny Ponomarev, Larisa Markeeva, Andrzej Cichocki, Ivan Oseledets",
    author_email="julgusak@gmail.com",
    url="https://github.com/musco-ai/musco-pytorch",
    download_url="https://github.com/musco-ai/musco-pytorch/archive/v1.0.1.tar.gz",
    license="Apache-2.0",
    packages=find_packages(),
    cmdclass={"install": InstallLocalPackage},
    install_requires=load_requirements("requirements.txt")
)
