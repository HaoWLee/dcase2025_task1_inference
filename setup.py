from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='Schmid_CPJKU_task1',
    version='0.1.0',
    description='Baseline Inference package',
    author='Florian Schmid',
    author_email="florian.schmid@jku.at",
    packages=find_packages(),  # This auto-discovers the inner folder
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'Schmid_CPJKU_task1': ["resources/*.wav", 'ckpts/*.ckpt'],
    },
    python_requires='>=3.13',
)
