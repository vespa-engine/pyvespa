import os
import setuptools


def get_target_version():
    pyvespa_version = os.environ.get("PYVESPA_VERSION", 0.7)
    build_nr = os.environ.get("SD_EVENT_ID", "0+dev")
    return "{}.{}".format(pyvespa_version, build_nr)


min_python = "3.6"

setuptools.setup(
    name="pyvespa",
    version=get_target_version(),
    description="Python API for vespa.ai",
    long_description="""[Vespa](https://vespa.ai/) is the scalable open-sourced serving engine
that enables users to store, compute and rank big data at user serving time.
[pyvespa](https://pyvespa.readthedocs.io/) provides a python API to Vespa.
It allows users to create, modify, deploy and interact with running Vespa instances.
The main goal of the library is to allow for faster prototyping
and to facilitate Machine Learning experiments for Vespa applications -
also see [learntorank](https://github.com/vespa-engine/learntorank).""",
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
        "pandas",
        "docker",
        "jinja2",
        "cryptography",
        "aiohttp",
        "tenacity",
    ],
    extras_require={
        "ml": ["transformers", "torch<1.13", "tensorflow", "tensorflow_ranking", "keras_tuner"],
        "full": ["onnxruntime", "transformers", "torch<1.13", "tensorflow", "tensorflow_ranking", "keras_tuner"],
    },
    python_requires=">=3.6",
    zip_safe=False,
    package_data={"vespa": ["py.typed"]},
    data_files=[
        (
            "templates",
            [
                "vespa/templates/hosts.xml",
                "vespa/templates/services.xml",
                "vespa/templates/schema.txt",
                "vespa/templates/query_profile.xml",
                "vespa/templates/query_profile_type.xml",
                "vespa/templates/validation-overrides.xml"
            ],
        )
    ],
)
