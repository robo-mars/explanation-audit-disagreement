from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class PreprocessorBundle:
    transformer: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]
    transformed_feature_names: list[str]
    transformed_to_original: list[str]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = self.transformer.transform(frame)
        return pd.DataFrame(transformed, index=frame.index, columns=self.transformed_feature_names)

    @property
    def original_feature_names(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


def fit_preprocessor(
    training_frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> PreprocessorBundle:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    transformer.fit(training_frame)

    transformed_feature_names: list[str] = []
    transformed_to_original: list[str] = []

    for feature_name in numeric_features:
        transformed_feature_names.append(feature_name)
        transformed_to_original.append(feature_name)

    if categorical_features:
        encoder: OneHotEncoder = transformer.named_transformers_["categorical"].named_steps["onehot"]
        for feature_name, categories in zip(categorical_features, encoder.categories_, strict=True):
            for category in categories:
                transformed_feature_names.append(f"{feature_name}={category}")
                transformed_to_original.append(feature_name)

    return PreprocessorBundle(
        transformer=transformer,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        transformed_feature_names=transformed_feature_names,
        transformed_to_original=transformed_to_original,
    )
