from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class CatBoostWithCatFeatures(BaseEstimator, ClassifierMixin):
    def __init__(self, cat_features=None, **catboost_params):
        self.cat_features = cat_features
        self.catboost_params = catboost_params
        self.model = CatBoostClassifier(**self.catboost_params)

    def fit(self, X, y):
        self.model.fit(X, y, cat_features=self.cat_features)
        # self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def get_catboost_pipeline(
    numerical_features: list[str], categorical_features: list[str]
):
    """
    Create a CatBoost pipeline with preprocessing steps.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    model = CatBoostWithCatFeatures(
        iterations=500,  # Number of boosting iterations
        learning_rate=0.2,  # Learning rate
        depth=6,  # Depth of the trees
        loss_function="MultiClass",  # Multi-class classification loss function
        verbose=100,  # Display progress every 100 iterations
        random_seed=42,  # Seed for reproducibility
        cat_features=categorical_features,
    )

    return Pipeline(
        [("preprocessor", preprocessor), ("classifier", model)],
        memory="../models_cache",
    )
