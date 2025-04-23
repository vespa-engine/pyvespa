from pathlib import Path
import os
from lxml import etree

# Dict of filename (without extension) to RELAXNG schema (open(rb))
RELAXNG = {}
to_import = ["services", "validation-overrides"]
for schema_file in Path(os.path.dirname(__file__)).glob("*.rng"):
    if schema_file.stem in to_import:
        with open(schema_file, "rb") as fh:
            RELAXNG[schema_file.stem] = etree.RelaxNG(etree.parse(fh))
