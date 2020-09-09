from vespa.package import Document, Field
from vespa.package import Schema, FieldSet, RankProfile
from vespa.package import ApplicationPackage
from vespa.package import VespaDocker

#
# Create application package
#
document = Document(
    fields=[
        Field(name="id", type="string", indexing=["attribute", "summary"]),
        Field(
            name="title",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
        Field(
            name="body",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
    ]
)
msmarco_schema = Schema(
    name="msmarco",
    document=document,
    fieldsets=[FieldSet(name="default", fields=["title", "body"])],
    rank_profiles=[RankProfile(name="default", first_phase="nativeRank(title, body)")],
)
app_package = ApplicationPackage(name="msmarco", schema=msmarco_schema)

#
# Deploy in a Docker container
#
vespa_docker = VespaDocker(application_package=app_package)
app = vespa_docker.deploy(disk_folder="sample_application")
# app.deployment_message
