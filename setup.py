import os
import setuptools


def get_target_version():
    pyvespa_version = os.environ.get("PYVESPA_VERSION", 0.7)
    build_nr = os.environ.get("SD_EVENT_ID", "0+dev")
    return "{}.{}".format(pyvespa_version, build_nr)


min_python = "3.8"

setuptools.setup(
    name="pyvespa",
    version=get_target_version(),
    description="Python API for vespa.ai",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://pyvespa.readthedocs.io/",
    keywords="vespa, search engine, data science",
    author="Thiago G. Martins",
    maintainer="Kristian Aune",
    maintainer_email="kraune@yahooinc.com",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "requests",
        "requests_toolbelt",
        "docker",
        "jinja2",
        "cryptography",
        "aiohttp",
        "tenacity",
        "typing_extensions",
    ],
    extras_require={
        "dev": [
            "pytest",
            "requests-mock",
            "datasets",
            "notebook",
            "nbconvert",
            "runnb",
        ],
    },
    python_requires=">=3.8",
    zip_safe=False,
    package_data={"vespa": ["py.typed"]},
    data_files=[
        (
            "templates",
            [
                "vespa/templates/hosts.xml",
                "vespa/templates/macros.txt",
                "vespa/templates/services.xml",
                "vespa/templates/schema.txt",
                "vespa/templates/query_profile.xml",
                "vespa/templates/query_profile_type.xml",
                "vespa/templates/validation-overrides.xml"
            ],
        )
    ],
)
