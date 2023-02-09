"""
In the preprocessing.py, add any abstractions required by the feature generation process.
"""

import functools

import pandas
import torch
from src import utils


def get_drug_cell_line_pairs(graph):
    df_expression = graph.query(
        f"""
        MATCH (c:CELL_LINE)-[r:HAS_GENE_OBSERVATION]->(g:GENE)
        WHERE  (c.tcga_code IN {str(utils.TCGA_label_set)}) AND (r.source = "genomics_expression")
        WITH c, r, g
        ORDER by g.ID
        RETURN c.CCLE_ID as CCLE_ID, collect(r.observation) as genomics_expression
        ORDER by c.CCLE_ID
        """
    )
    df_expression = pandas.concat(df_expression)

    df_mutation = graph.query(
        f"""
        MATCH (c:CELL_LINE)-[r:HAS_GENE_OBSERVATION]->(g:GENE)
        WHERE (c.tcga_code IN {str(utils.TCGA_label_set)}) AND (r.source = "genomics_mutation")
        WITH c, r, g
        ORDER by g.ID
        RETURN c.CCLE_ID as CCLE_ID, collect(r.observation) as genomics_mutation
        ORDER by c.CCLE_ID
        """
    )
    df_mutation = pandas.concat(df_mutation)

    df_methylation = graph.query(
        f"""
        MATCH (c:CELL_LINE)-[r:HAS_GENE_OBSERVATION]->(g:GENE)
        WHERE (c.tcga_code IN {str(utils.TCGA_label_set)}) AND (r.source = "genomics_methylation")
        WITH c, r, g
        ORDER by g.ID
        RETURN c.CCLE_ID as CCLE_ID, collect(r.observation) as genomics_methylation
        ORDER by c.CCLE_ID
        """
    )
    df_methylation = pandas.concat(df_methylation)

    df_cell_lines = functools.reduce(
        lambda left, right: pandas.merge(left, right, on=["CCLE_ID"]), [df_expression, df_methylation, df_mutation]
    )
    df_cell_lines = df_cell_lines.set_index("CCLE_ID")

    assert torch.tensor(df_cell_lines.iloc[0].genomics_expression).shape[0] == 697
    assert torch.tensor(df_cell_lines.iloc[0].genomics_methylation).shape[0] == 808
    assert torch.tensor(df_cell_lines.iloc[0].genomics_mutation).shape[0] == 34673

    df_target = graph.query(
        f"""MATCH (a:DRUG)<-[:FOR_DRUG]-(:GDSC)-[u:HAS_CELL_LINE]->(c:CELL_LINE)
                    WHERE c.tcga_code IN {str(utils.TCGA_label_set)}
                    RETURN a.smiles as smiles, c.CCLE_ID as CCLE_ID, u.observation as label, c.tcga_code as tcga_code
                    ORDER BY smiles, CCLE_ID"""
    )
    df_target = pandas.concat(df_target)
    df_target = df_target[df_target["CCLE_ID"].isin(df_cell_lines.index.values.tolist())]
    df_target.dropna(inplace=True)
    df_target.reset_index(drop=True, inplace=True)
    return df_cell_lines, df_target
