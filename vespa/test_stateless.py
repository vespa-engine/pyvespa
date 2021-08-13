import unittest

from vespa.stateless import ModelServer


class TestModelServer(unittest.TestCase):
    def setUp(self) -> None:
        self.server_name = "test_server"
        self.model_server = ModelServer(
            name=self.server_name,
        )

    def test_model_server_serialization(self):
        self.assertEqual(
            self.model_server, ModelServer.from_dict(self.model_server.to_dict)
        )

    def test_get_schema(self):
        self.assertIsNone(self.model_server.schema)
        self.assertEqual(self.model_server.schema, self.model_server.get_schema())

    def test_hosts_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="utf-8" ?>\n'
            "<!-- Copyright 2019 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root. -->\n"
            "<hosts>\n"
            '    <host name="localhost">\n'
            "        <alias>node1</alias>\n"
            "    </host>\n"
            "</hosts>"
        )
        self.assertEqual(self.model_server.hosts_to_text, expected_result)

    def test_services_to_text(self):
        expected_result = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<services version="1.0">\n'
            '    <container id="test_server_container" version="1.0">\n'
            "        <model-evaluation/>\n"
            "    </container>\n"
            "</services>"
        )

        self.assertEqual(self.model_server.services_to_text, expected_result)

    def test_query_profile_to_text(self):
        expected_result = (
            '<query-profile id="default" type="root">\n' "</query-profile>"
        )
        self.assertEqual(self.model_server.query_profile_to_text, expected_result)

    def test_query_profile_type_to_text(self):
        expected_result = '<query-profile-type id="root">\n' "</query-profile-type>"
        self.assertEqual(self.model_server.query_profile_type_to_text, expected_result)
