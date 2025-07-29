# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest

from vespa.package import Field, Document, Schema, ApplicationPackage, HNSW


class TestFieldMultipleIndex(unittest.TestCase):
    """Tests for the multiple index configurations feature in Field definitions."""

    def test_single_string_index_backward_compatibility(self):
        """Test that single string index configuration works as before."""
        field = Field(name="title", type="string", index="enable-bm25")
        self.assertEqual(field.index, "enable-bm25")
        self.assertEqual(field.index_configurations, ["enable-bm25"])

    def test_single_dict_index_backward_compatibility(self):
        """Test that single dict index configuration works as before."""
        index_config = {"arity": 2, "lower-bound": 3}
        field = Field(name="predicate_field", type="predicate", index=index_config)
        self.assertEqual(field.index, index_config)
        self.assertEqual(field.index_configurations, [index_config])

    def test_no_index_configuration(self):
        """Test field with no index configuration."""
        field = Field(name="no_index", type="string")
        self.assertIsNone(field.index)
        self.assertEqual(field.index_configurations, [])

    def test_multiple_string_indices(self):
        """Test field with multiple string index configurations."""
        field = Field(name="multi_string", type="string", index=["enable-bm25", "another-setting"])
        self.assertEqual(field.index, ["enable-bm25", "another-setting"])
        self.assertEqual(field.index_configurations, ["enable-bm25", "another-setting"])

    def test_multiple_dict_indices(self):
        """Test field with multiple dict index configurations."""
        indices = [{"param1": "value1"}, {"param2": "value2"}]
        field = Field(name="multi_dict", type="string", index=indices)
        self.assertEqual(field.index, indices)
        self.assertEqual(field.index_configurations, indices)

    def test_mixed_string_and_dict_indices(self):
        """Test field with mixed string and dict index configurations."""
        indices = ["enable-bm25", {"arity": 2}, "another-setting"]
        field = Field(name="mixed", type="string", index=indices)
        self.assertEqual(field.index, indices)
        self.assertEqual(field.index_configurations, indices)

    def test_predicate_field_from_issue(self):
        """Test the exact predicate field example from issue #983."""
        field = Field(
            name="predicate_field",
            type="predicate",
            indexing=["attribute"],
            index={
                "arity": 2,
                "lower-bound": 3,
                "upper-bound": 200,
                "dense-posting-list-threshold": 0.25
            }
        )
        
        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text
        
        # Verify all expected elements are present
        self.assertIn("field predicate_field type predicate", schema_text)
        self.assertIn("indexing: attribute", schema_text)
        self.assertIn("arity: 2", schema_text)
        self.assertIn("lower-bound: 3", schema_text)
        self.assertIn("upper-bound: 200", schema_text)
        self.assertIn("dense-posting-list-threshold: 0.25", schema_text)

    def test_multiple_index_configurations_rendering(self):
        """Test that multiple index configurations render correctly in schema."""
        field = Field(
            name="multi_index",
            type="string",
            indexing=["index", "summary"],
            index=["enable-bm25", {"arity": 2, "lower-bound": 3}, "another-setting"]
        )
        
        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text
        
        # Should have multiple index statements
        self.assertIn("index: enable-bm25", schema_text)
        self.assertIn("index: another-setting", schema_text)
        self.assertIn("arity: 2", schema_text)
        self.assertIn("lower-bound: 3", schema_text)
        
        # Count occurrences of "index" to ensure multiple index blocks
        index_count = schema_text.count("index:")
        index_block_count = schema_text.count("index {")
        self.assertGreaterEqual(index_count + index_block_count, 3)  # At least 3 index-related entries

    def test_dict_with_none_values(self):
        """Test that dict index configurations with None values render without ': None'."""
        field = Field(
            name="parameterless",
            type="string",
            index={"enable-bm25": None, "param": "value"}
        )
        
        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text
        
        # Should have enable-bm25 without ": None" and param with value
        self.assertIn("enable-bm25\n", schema_text.replace(" ", "").replace("\t", ""))
        self.assertIn("param: value", schema_text)
        self.assertNotIn("enable-bm25: None", schema_text)

    def test_ann_field_with_additional_index_configs(self):
        """Test that ANN fields work correctly with additional index configurations."""
        field = Field(
            name="vector_field",
            type="tensor<float>(x[128])",
            indexing=["attribute"],
            ann=HNSW(
                distance_metric="euclidean",
                max_links_per_node=16,
                neighbors_to_explore_at_insert=200,
            ),
            index=["enable-bm25", {"custom-param": "value"}]
        )
        
        # Create schema and render
        document = Document(fields=[field])
        schema = Schema("test", document)
        schema_text = schema.schema_to_text
        
        # Should have both ANN index block and custom index configurations
        self.assertIn("index: enable-bm25", schema_text)
        self.assertIn("custom-param: value", schema_text)
        self.assertIn("hnsw {", schema_text)
        self.assertIn("max-links-per-node: 16", schema_text)
        self.assertIn("distance-metric: euclidean", schema_text)

    def test_equality_with_multiple_indices(self):
        """Test that Field equality works correctly with multiple index configurations."""
        index_config = ["enable-bm25", {"arity": 2}]
        field1 = Field(name="test", type="string", index=index_config)
        field2 = Field(name="test", type="string", index=index_config.copy())
        field3 = Field(name="test", type="string", index="enable-bm25")
        
        self.assertEqual(field1, field2)
        self.assertNotEqual(field1, field3)


if __name__ == "__main__":
    unittest.main()