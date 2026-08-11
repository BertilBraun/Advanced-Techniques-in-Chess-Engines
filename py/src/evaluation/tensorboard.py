from __future__ import annotations

import re

from src.evaluation.configuration import EvaluationConfiguration
from src.util.tensorboard import TensorboardCustomScalarCategory, TensorboardMultilineChart


EVALUATION_TAG_PATTERN = re.compile(r'^((?:coordinator/)?evaluation(?:_metadata)?)/([^/]+)/([^/]+)$')
MATCH_OUTCOME_METRICS = ('wins', 'draws', 'losses')
MATCH_SCORE_METRICS = ('score',)
MATCH_METADATA_METRICS = ('first_player_score', 'second_player_score')
FIXED_DATASET_METRICS = ('top_action_accuracy', 'policy_cross_entropy')


def _categories_from_metrics(
    metrics_by_definition: dict[tuple[str, str], set[str]],
) -> tuple[TensorboardCustomScalarCategory, ...]:
    match_charts: list[TensorboardMultilineChart] = []
    for (evaluation_prefix, definition_id), metrics in sorted(metrics_by_definition.items()):
        tag_prefix = f'{evaluation_prefix}/{definition_id}'
        if set(MATCH_OUTCOME_METRICS).issubset(metrics):
            match_charts.append(
                TensorboardMultilineChart(
                    title=f'{definition_id} W/D/L',
                    tags=tuple(f'{tag_prefix}/{metric}' for metric in MATCH_OUTCOME_METRICS),
                )
            )
        if set(MATCH_SCORE_METRICS).issubset(metrics):
            match_charts.append(
                TensorboardMultilineChart(
                    title=f'{definition_id} score',
                    tags=tuple(f'{tag_prefix}/{metric}' for metric in MATCH_SCORE_METRICS),
                )
            )

    dataset_charts: list[TensorboardMultilineChart] = []
    for metric, title in (
        ('top_action_accuracy', 'Top-action accuracy'),
        ('policy_cross_entropy', 'Policy cross-entropy'),
    ):
        tags = tuple(
            f'{evaluation_prefix}/{definition_id}/{metric}'
            for (evaluation_prefix, definition_id), metrics in sorted(metrics_by_definition.items())
            if metric in metrics
        )
        if tags:
            dataset_charts.append(TensorboardMultilineChart(title=title, tags=tags))

    metadata_charts: list[TensorboardMultilineChart] = []
    for (evaluation_prefix, definition_id), metrics in sorted(metrics_by_definition.items()):
        if set(MATCH_METADATA_METRICS).issubset(metrics):
            metadata_charts.append(
                TensorboardMultilineChart(
                    title=f'{definition_id} player scores',
                    tags=tuple(f'{evaluation_prefix}/{definition_id}/{metric}' for metric in MATCH_METADATA_METRICS),
                )
            )
    duration_tags = tuple(
        f'{evaluation_prefix}/{definition_id}/duration_seconds'
        for (evaluation_prefix, definition_id), metrics in sorted(metrics_by_definition.items())
        if 'duration_seconds' in metrics
    )
    categories: list[TensorboardCustomScalarCategory] = []
    if match_charts:
        categories.append(TensorboardCustomScalarCategory(title='Evaluation matches', charts=tuple(match_charts)))
    if dataset_charts:
        categories.append(TensorboardCustomScalarCategory(title='Evaluation datasets', charts=tuple(dataset_charts)))
    if duration_tags:
        metadata_charts.append(TensorboardMultilineChart(title='Duration (seconds)', tags=duration_tags))
    if metadata_charts:
        categories.append(
            TensorboardCustomScalarCategory(
                title='Evaluation metadata',
                charts=tuple(metadata_charts),
            )
        )
    return tuple(categories)


def evaluation_tensorboard_categories(
    configuration: EvaluationConfiguration,
) -> tuple[TensorboardCustomScalarCategory, ...]:
    metrics_by_definition: dict[tuple[str, str], set[str]] = {}
    for definition in configuration.definitions:
        metadata_metrics = {'duration_seconds'}
        if definition.kind == 'fixed_dataset':
            metrics = set(FIXED_DATASET_METRICS)
        else:
            metrics = {*MATCH_OUTCOME_METRICS, *MATCH_SCORE_METRICS}
            metadata_metrics.update(MATCH_METADATA_METRICS)
        metrics_by_definition[('evaluation', definition.definition_id)] = metrics
        metrics_by_definition[('evaluation_metadata', definition.definition_id)] = metadata_metrics
    return _categories_from_metrics(metrics_by_definition)


def discovered_evaluation_tensorboard_categories(
    scalar_tags: set[str],
) -> tuple[TensorboardCustomScalarCategory, ...]:
    metrics_by_definition: dict[tuple[str, str], set[str]] = {}
    for tag in scalar_tags:
        match = EVALUATION_TAG_PATTERN.fullmatch(tag)
        if match is None:
            continue
        evaluation_prefix, definition_id, metric = match.groups()
        metrics_by_definition.setdefault((evaluation_prefix, definition_id), set()).add(metric)
    return _categories_from_metrics(metrics_by_definition)
