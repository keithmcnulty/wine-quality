from kedro.pipeline import node, Pipeline

from .nodes import split_data, scale_data, generate_classification_report, train_logit, train_knn_crossvalidated, train_xgb_crossvalidated

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["all_wine_dummy", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="splitting_data",
            ),
            node(
                func=scale_data,
                inputs=["X_train", "X_test"],
                outputs=["X_train_scaled", "X_test_scaled"],
                name="scaling_data"
            ),
            node(
                func=train_logit,
                inputs=["X_train", "y_train"],
                outputs="logit",
                name="Training_logit_model",
            ),
            node(
                func=generate_classification_report,
                inputs=["logit", "X_test", "y_test"],
                outputs=None,
                name="logging_performance_of_logit",
            ),
            node(
                func=train_knn_crossvalidated,
                inputs=["X_train_scaled", "y_train", "parameters"],
                outputs="knn",
                name="Training_knn_model",
            ),
            node(
                func=generate_classification_report,
                inputs=["knn", "X_test_scaled", "y_test"],
                outputs=None,
                name="logging_performance_of_knn",
            ),
            node(
                func=train_xgb_crossvalidated,
                inputs=["X_train_scaled", "y_train", "parameters"],
                outputs="xgb",
                name="Training_xgb_model",
            ),
            node(
                func=generate_classification_report,
                inputs=["xgb", "X_test_scaled", "y_test"],
                outputs=None,
                name="logging_performance_of_xgb",
            ),
        ]
    )