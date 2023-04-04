"""
In the preprocessing.py, add any abstractions required by the feature generation process.
"""


def set_cell_line_property(graph, edge_label: str, genomics_property: str):
    graph.query(
        f"""
        MATCH (c:CELL_LINE)-[r:{edge_label}]->(g:GENE)
        WITH c, r, g
        ORDER by g.ID
        WITH c, collect(r.observation) as {genomics_property}
        SET c.{genomics_property} = {genomics_property}
        """
    )


def delete_single_node(graph):
    """Delete single node in a graph"""
    graph.query(
        """
        MATCH (n)
        WHERE NOT (n)--()
        DELETE n
        """
    )


def remove_null_cells(graph):
    """Delete cells node with NULL genomics features"""
    graph.query(
        """
        MATCH (c:CELL_LINE)
        WHERE c.genomics_methylation IS NULL
            OR c.genomics_expression IS NULL
            OR c.genomics_mutation IS NULL
        DETACH DELETE c
        """
    )
