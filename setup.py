from setuptools import setup, find_packages
import glob

setup(
    name='kpi-mpc',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    scripts=glob.glob('src/scripts/*.py'),
    url='https://gitlab.ifremer.fr/lops-wave/kpi_mpc',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    install_requires=[
        'xarray',
        'netCDF4',
        'numpy',
        'scipy',
    ],
    entry_points={
    },
    license='MIT',
    author='Antoine Grouazel',
    author_email='Antoine.grouazel@ifremer.fr',
    description='libraries to compute Key Performance Indicators for Mission Performance Center (Sentinel-1 SAR mission)'
)
