import json
import glob
import re
import pprint

"""
{
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "additionalProperties": false,
    "properties": {
        "ReferenceToLocalSchema": {
            "$ref": "#/definitions/LocalType"
        },
        "ReferenceToExternalSchema": {
            "$ref": "Common.json#/definitions/ExternalType"
        }
    },
    "definitions": {
        "LocalType": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "no-write": {
                    "type": "boolean",
                    "default": false
                }
            }
        }
    }
}
"""

def fix_definitions(content):
    return re.sub("#/definitions/([a-zA-Z]+)", "common.json#/definitions/\\1", content)

common_schemas = {"definitions": {}}

for fn in glob.glob("schemas_raw/*.json"):
    with open(fn) as inp:
        fixed_definition = json.loads(fix_definitions(inp.read()))
    # pprint.pprint(fixed_definition)
    if "definitions" in fixed_definition:
        for key in fixed_definition["definitions"]:
            print(key)
            common_schemas["definitions"][key] = fixed_definition["definitions"][key].copy()
        del fixed_definition["definitions"]

    # this doesn't work - default value is not respected by quicktype
    # if "properties" in fixed_definition and "event_type" in fixed_definition["properties"]:
    #     fixed_definition["properties"]["event_type"]["default"] = fixed_definition["title"]

    filename = fn.split("/")[-1]
    with open("schemas/{}".format(filename),"wt") as out:
        out.write(json.dumps(fixed_definition, indent=4))


with open("schemas/common.json", "wt") as out:
    out.write(json.dumps(common_schemas, indent=4))
