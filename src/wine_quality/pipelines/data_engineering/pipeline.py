from kedro.pipeline import node, Pipeline

from .nodes import quality_category, concat_dfs, create_dummies

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=quality_category,
                inputs="white_wine",
                outputs="white_wine_categorized",
                name="categorizing_white_wine",
            ),
            node(
                func=quality_category,
                inputs="red_wine",
                outputs="red_wine_categorized",
                name="categorizing_red_wine",
            ),
            node(
                func=concat_dfs,
                inputs=["red_wine_categorized", "white_wine_categorized"],
                outputs="all_wine",
                name="concatenating_wines",
            ),
            node(
                func=create_dummies,
                inputs=["all_wine", "parameters"],
                outputs="all_wine_dummy",
                name="creating_dummy_variables",
            ),
        ]
    )