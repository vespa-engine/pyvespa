import os
import setuptools


def get_target_version():
    pyvespa_version = os.environ.get("PYVESPA_VERSION", "0.2")
    build_nr = os.environ.get("TRAVIS_BUILD_NUMBER", "0+dev")
    return "{}.{}".format(pyvespa_version, build_nr)


min_python = "3.6"

setuptools.setup(
    name="pyvespa",
    version=get_target_version(),
    description="Python API for vespa.ai",
    keywords="vespa, search engine, data science",
    author="Thiago G. Martins",
    author_email="tmartins@verizonmedia.com",
    license=(
        "Apache Software License 2.0",
        "OSI Approved :: Apache Software License",
    ),
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "requests",
        "pandas",
        "docker",
        "jinja2",
        "cryptography",
    ],
    extras_require={
        "ml": ["transformers", "torch"],
        "full": ["transformers", "torch", "onnxruntime"],
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
