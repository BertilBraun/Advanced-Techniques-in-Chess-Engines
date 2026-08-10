from __future__ import annotations

import re

from src.evaluation.configuration import EvaluationConfiguration
from src.util.tensorboard import TensorboardCustomScalarCategory, TensorboardMultilineChart


EVALUATION_TAG_PATTERN = re.compile(r'^((?:coordinator/)?evaluation)/([^/]+)/([^/]+)$')
MATCH_OUTCOME_METRICS = ('wins', 'draws', 'losses')
MATCH_SCORE_METRICS = ('score', 'first_player_score', 'second_player_score')
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
                    title=f'{definition_id} scores',
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
        categories.append(
            TensorboardCustomScalarCategory(
                title='Evaluation timing',
                charts=(TensorboardMultilineChart(title='Duration (seconds)', tags=duration_tags),),
            )
        )
    return tuple(categories)


def evaluation_tensorboard_categories(
    configuration: EvaluationConfiguration,
) -> tuple[TensorboardCustomScalarCategory, ...]:
    metrics_by_definition: dict[tuple[str, str], set[str]] = {}
    for definition in configuration.definitions:
        metrics = {'duration_seconds'}
        if definition.kind == 'fixed_dataset':
            metrics.update(FIXED_DATASET_METRICS)
        else:
            metrics.update(MATCH_OUTCOME_METRICS)
            metrics.update(MATCH_SCORE_METRICS)
        metrics_by_definition[('evaluation', definition.definition_id)] = metrics
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
