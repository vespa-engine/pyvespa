import os
import setuptools


def get_target_version():
    pyvespa_version = os.environ.get("PYVESPA_VERSION", 0.7)
    build_nr = os.environ.get("SD_EVENT_ID", "0+dev")
    return "{}.{}".format(pyvespa_version, build_nr)


min_python = "3.6"

setuptools.setup(
    name="pyvespa",
    version="0.19.0",
    description="Python API for vespa.ai",
    keywords="vespa, search engine, data science",
    author="Thiago G. Martins",
    author_email="tmartins@verizonmedia.com",
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
        "ml": ["transformers", "torch", "tensorflow", "tensorflow_ranking", "keras_tuner"],
        "full": ["onnxruntime", "transformers", "torch", "tensorflow", "tensorflow_ranking", "keras_tuner"],
    },
    python_requires=">=3.6",
    zip_safe=False,
    data_files=[
        (
            "templates",
            [
                "vespa/templates/hosts.xml",
                "vespa/templates/services.xml",
                "vespa/templates/schema.txt",
                "vespa/templates/query_profile.xml",
                "vespa/templates/query_profile_type.xml",
            ],
        )
    ],
)
