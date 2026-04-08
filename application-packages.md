# Application packages[¶](#application-packages)

Vespa is configured using an [application package](https://docs.vespa.ai/en/application-packages.html). Pyvespa provides an API to generate a deployable application package. An application package has at a minimum a [schema](https://docs.vespa.ai/en/schemas.html) and [services.xml](https://docs.vespa.ai/en/reference/services.html).

> ***NOTE: pyvespa generally does not support all indexing options in Vespa - it is made for easy experimentation.*** ***To configure setting an unsupported indexing option (or any other unsupported option),*** ***export the application package like above, modify the schema or other files*** ***and deploy the application package from the directory, or as a zipped file.*** ***Find more details at the end of this notebook.***

Refer to [troubleshooting](https://vespa-engine.github.io/pyvespa/troubleshooting.md) for any problem when running this guide.

In \[ \]:

Copied!

```
!pip3 install pyvespa
```

!pip3 install pyvespa

By exporting to disk, one can see the generated files:

In \[50\]:

Copied!

```
import os
import tempfile
from pathlib import Path
from vespa.package import ApplicationPackage

app_name = "myschema"
app_package = ApplicationPackage(name=app_name, create_query_profile_by_default=False)

temp_dir = tempfile.TemporaryDirectory()
app_package.to_files(temp_dir.name)

for p in Path(temp_dir.name).rglob("*"):
    if p.is_file():
        print(p)
```

import os import tempfile from pathlib import Path from vespa.package import ApplicationPackage app_name = "myschema" app_package = ApplicationPackage(name=app_name, create_query_profile_by_default=False) temp_dir = tempfile.TemporaryDirectory() app_package.to_files(temp_dir.name) for p in Path(temp_dir.name).rglob("\*"): if p.is_file(): print(p)

```
/var/folders/9_/z105jyln7jz8h2vwsrjb7kxh0000gp/T/tmp6geo2dpg/services.xml
/var/folders/9_/z105jyln7jz8h2vwsrjb7kxh0000gp/T/tmp6geo2dpg/schemas/myschema.sd
```

## Schema[¶](#schema)

A schema is created with the same name as the application package:

In \[51\]:

Copied!

```
os.environ["TMP_APP_DIR"] = temp_dir.name
os.environ["APP_NAME"] = app_name

!cat $TMP_APP_DIR/schemas/$APP_NAME.sd
```

os.environ["TMP_APP_DIR"] = temp_dir.name os.environ["APP_NAME"] = app_name !cat $TMP_APP_DIR/schemas/$APP_NAME.sd

```
schema myschema {
    document myschema {
    }
}
```

Configure the schema with [fields](https://docs.vespa.ai/en/schemas.html#field), [fieldsets](https://docs.vespa.ai/en/schemas.html#fieldset) and a [ranking function](https://docs.vespa.ai/en/ranking.html):

In \[52\]:

Copied!

```
from vespa.package import Field, FieldSet, RankProfile

app_package.schema.add_fields(
    Field(name="id", type="string", indexing=["attribute", "summary"]),
    Field(
        name="title", type="string", indexing=["index", "summary"], index="enable-bm25"
    ),
    Field(
        name="body", type="string", indexing=["index", "summary"], index="enable-bm25"
    ),
)

app_package.schema.add_field_set(FieldSet(name="default", fields=["title", "body"]))

app_package.schema.add_rank_profile(
    RankProfile(name="default", first_phase="bm25(title) + bm25(body)")
)
```

from vespa.package import Field, FieldSet, RankProfile app_package.schema.add_fields( Field(name="id", type="string", indexing=["attribute", "summary"]), Field( name="title", type="string", indexing=["index", "summary"], index="enable-bm25" ), Field( name="body", type="string", indexing=["index", "summary"], index="enable-bm25" ), ) app_package.schema.add_field_set(FieldSet(name="default", fields=["title", "body"])) app_package.schema.add_rank_profile( RankProfile(name="default", first_phase="bm25(title) + bm25(body)") )

Export the application package again, show schema:

In \[53\]:

Copied!

```
app_package.to_files(temp_dir.name)

!cat $TMP_APP_DIR/schemas/$APP_NAME.sd
```

app_package.to_files(temp_dir.name) !cat $TMP_APP_DIR/schemas/$APP_NAME.sd

```
schema myschema {
    document myschema {
        field id type string {
            indexing: attribute | summary
        }
        field title type string {
            indexing: index | summary
            index: enable-bm25
        }
        field body type string {
            indexing: index | summary
            index: enable-bm25
        }
    }
    fieldset default {
        fields: title, body
    }
    rank-profile default {
        first-phase {
            expression {
                bm25(title) + bm25(body)
            }
        }
    }
}
```

## Services[¶](#services)

`services.xml` configures container and content clusters - see the [Vespa Overview](https://docs.vespa.ai/en/overview.html). This is a file you will normally not change or need to know much about:

In \[54\]:

Copied!

```
!cat $TMP_APP_DIR/services.xml
```

!cat $TMP_APP_DIR/services.xml

```
<?xml version="1.0" encoding="UTF-8"?>
<services version="1.0">
    <container id="myschema_container" version="1.0">
        <search></search>
        <document-api></document-api>
    </container>
    <content id="myschema_content" version="1.0">
        <redundancy reply-after="1">1</redundancy>
        <documents>
            <document type="myschema" mode="index"></document>
        </documents>
        <nodes>
            <node distribution-key="0" hostalias="node1"></node>
        </nodes>
    </content>
</services>
```

Observe:

- A *content cluster* (this is where the index is stored) called `myschema_content` is created. This is information not normally needed, unless using [delete_all_docs](https://vespa-engine.github.io/pyvespa/api/vespa/application.md#vespa.application.Vespa.delete_all_docs) to quickly remove all documents from a schema

## Deploy[¶](#deploy)

After completing the code for the fields and ranking, deploy the application into a Docker container - the container is started by pyvespa:

In \[55\]:

Copied!

```
from vespa.deployment import VespaDocker

vespa_container = VespaDocker()
vespa_connection = vespa_container.deploy(application_package=app_package)
```

from vespa.deployment import VespaDocker vespa_container = VespaDocker() vespa_connection = vespa_container.deploy(application_package=app_package)

```
Waiting for configuration server, 0/300 seconds...
Waiting for configuration server, 5/300 seconds...
Waiting for application status, 0/300 seconds...
Waiting for application status, 5/300 seconds...
Waiting for application status, 10/300 seconds...
Waiting for application status, 15/300 seconds...
Waiting for application status, 20/300 seconds...
Waiting for application status, 25/300 seconds...
Finished deployment.
```

## Deploy from modified files[¶](#deploy-from-modified-files)

To add configuration the the schema, which is not supported by the pyvespa code, export the files, modify, then deploy by using `deploy_from_disk`. This example adds custom configuration to the `services.xml` file above and deploys it:

In \[56\]:

Copied!

```
%%sh
cat << EOF > $TMP_APP_DIR/services.xml
<?xml version="1.0" encoding="UTF-8"?>
<services version="1.0">
    <container id="${APP_NAME}_container" version="1.0">
        <search></search>
        <document-api></document-api>
    </container>
    <content id="${APP_NAME}_content" version="1.0">
        <redundancy reply-after="1">1</redundancy>
        <documents>
            <document type="${APP_NAME}" mode="index"></document>
        </documents>
        <nodes>
            <node distribution-key="0" hostalias="node1"></node>
        </nodes>
        <tuning>
            <resource-limits>
                <disk>0.90</disk>
            </resource-limits>
        </tuning>
    </content>
</services>
EOF
```

%%sh cat \<< EOF > $TMP_APP_DIR/services.xml

<?xml version="1.0" encoding="UTF-8"?>

<services version="1.0">
<container id="${APP_NAME}_container" version="1.0">
<search></search>
<document-api></document-api>
</container>
<content id="${APP_NAME}_content" version="1.0">
<redundancy reply-after="1">1</redundancy>
<documents>
<document type="${APP_NAME}" mode="index"></document>
</documents>
<nodes>
<node distribution-key="0" hostalias="node1"></node>
</nodes>
<tuning>
<resource-limits>
<disk>0.90</disk>
</resource-limits>
</tuning>
</content>
</services>
EOF

The [resource-limits](https://docs.vespa.ai/en/reference/services-content.html#resource-limits) in `tuning/resource-limits/disk` configuration setting allows a higher disk usage.

Deploy using the exported files:

In \[57\]:

Copied!

```
vespa_connection = vespa_container.deploy_from_disk(
    application_name=app_name, application_root=temp_dir.name
)
```

vespa_connection = vespa_container.deploy_from_disk( application_name=app_name, application_root=temp_dir.name )

```
Waiting for configuration server, 0/300 seconds...
Waiting for configuration server, 5/300 seconds...
Waiting for application status, 0/300 seconds...
Waiting for application status, 5/300 seconds...
Finished deployment.
```

One can also export a deployable zip-file, which can be deployed using the Vespa Cloud Console:

In \[58\]:

Copied!

```
Path.mkdir(Path(temp_dir.name) / "zip", exist_ok=True, parents=True)
app_package.to_zipfile(temp_dir.name + "/zip/application.zip")

! find "$TMP_APP_DIR/zip" -type f
```

Path.mkdir(Path(temp_dir.name) / "zip", exist_ok=True, parents=True) app_package.to_zipfile(temp_dir.name + "/zip/application.zip") ! find "$TMP_APP_DIR/zip" -type f

```
/var/folders/9_/z105jyln7jz8h2vwsrjb7kxh0000gp/T/tmp6geo2dpg/zip/application.zip
```

### Cleanup[¶](#cleanup)

Remove the container resources and temporary application package file export:

In \[59\]:

Copied!

```
temp_dir.cleanup()
vespa_container.container.stop()
vespa_container.container.remove()
```

temp_dir.cleanup() vespa_container.container.stop() vespa_container.container.remove()

## Next step: Deploy, feed and query[¶](#next-step-deploy-feed-and-query)

Once the schema is ready for deployment, decide deployment option and deploy the application package:

- [Deploy to local container](https://vespa-engine.github.io/pyvespa/getting-started-pyvespa.md)
- [Deploy to Vespa Cloud](https://vespa-engine.github.io/pyvespa/getting-started-pyvespa-cloud.md)

Use the guides on the pyvespa site to feed and query data.
