from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from src.self_play.value_target import FinalOutcome, TerminationReason
from src.util.tensorboard import log_scalar


EXPECTED_SCORE_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class ValueMetrics:
    outcome_cross_entropy_sum: float = 0.0
    brier_score_sum: float = 0.0
    expected_score_mse_sum: float = 0.0
    expected_score_mae_sum: float = 0.0
    predicted_expected_score_sum: float = 0.0
    target_expected_score_sum: float = 0.0
    outcome_target_count: int = 0
    mcts_huber_sum: float = 0.0
    mcts_target_count: int = 0
    class_probability_sums: tuple[float, float, float] = (0.0, 0.0, 0.0)
    class_correct_counts: tuple[int, int, int] = (0, 0, 0)
    class_target_counts: tuple[int, int, int] = (0, 0, 0)
    expected_score_bin_prediction_sums: tuple[float, ...] = (0.0,) * EXPECTED_SCORE_CALIBRATION_BINS
    expected_score_bin_target_sums: tuple[float, ...] = (0.0,) * EXPECTED_SCORE_CALIBRATION_BINS
    expected_score_bin_counts: tuple[int, ...] = (0,) * EXPECTED_SCORE_CALIBRATION_BINS

    def __post_init__(self) -> None:
        bin_lengths = {
            len(self.expected_score_bin_prediction_sums),
            len(self.expected_score_bin_target_sums),
            len(self.expected_score_bin_counts),
        }
        if bin_lengths != {EXPECTED_SCORE_CALIBRATION_BINS}:
            raise ValueError(f'Expected-score calibration requires {EXPECTED_SCORE_CALIBRATION_BINS} bins.')

    @property
    def outcome_cross_entropy(self) -> float:
        return self.outcome_cross_entropy_sum / self.outcome_target_count if self.outcome_target_count else 0.0

    @property
    def brier_score(self) -> float:
        return self.brier_score_sum / self.outcome_target_count if self.outcome_target_count else 0.0

    @property
    def expected_score_mse(self) -> float:
        return self.expected_score_mse_sum / self.outcome_target_count if self.outcome_target_count else 0.0

    @property
    def expected_score_mae(self) -> float:
        return self.expected_score_mae_sum / self.outcome_target_count if self.outcome_target_count else 0.0

    @property
    def expected_score_bias(self) -> float:
        if not self.outcome_target_count:
            return 0.0
        return (self.predicted_expected_score_sum - self.target_expected_score_sum) / self.outcome_target_count

    @property
    def mcts_huber(self) -> float:
        return self.mcts_huber_sum / self.mcts_target_count if self.mcts_target_count else 0.0

    @property
    def expected_score_calibration_error(self) -> float:
        if not self.outcome_target_count:
            return 0.0
        absolute_error_sum = sum(
            abs(prediction_sum / count - target_sum / count) * count
            for prediction_sum, target_sum, count in zip(
                self.expected_score_bin_prediction_sums,
                self.expected_score_bin_target_sums,
                self.expected_score_bin_counts,
            )
            if count
        )
        return absolute_error_sum / self.outcome_target_count

    def combined_value_loss(self, outcome_weight: float, mcts_weight: float) -> float:
        return outcome_weight * self.outcome_cross_entropy + mcts_weight * self.mcts_huber

    def log_to_tensorboard(self, prefix: str, step: int) -> None:
        log_scalar(f'{prefix}/wdl_cross_entropy', self.outcome_cross_entropy, step)
        log_scalar(f'{prefix}/wdl_brier', self.brier_score, step)
        log_scalar(f'{prefix}/expected_score_mse', self.expected_score_mse, step)
        log_scalar(f'{prefix}/expected_score_mae', self.expected_score_mae, step)
        log_scalar(f'{prefix}/expected_score_bias', self.expected_score_bias, step)
        log_scalar(
            f'{prefix}/expected_score_calibration_error',
            self.expected_score_calibration_error,
            step,
        )
        log_scalar(f'{prefix}/mcts_huber', self.mcts_huber, step)
        log_scalar(f'{prefix}/outcome_target_count', self.outcome_target_count, step)
        log_scalar(f'{prefix}/mcts_target_count', self.mcts_target_count, step)
        for outcome in FinalOutcome:
            class_index = int(outcome)
            class_count = self.class_target_counts[class_index]
            if class_count == 0:
                continue
            class_prefix = f'{prefix}/wdl_class/{outcome.name.lower()}'
            log_scalar(
                f'{class_prefix}_mean_probability',
                self.class_probability_sums[class_index] / class_count,
                step,
            )
            log_scalar(
                f'{class_prefix}_accuracy',
                self.class_correct_counts[class_index] / class_count,
                step,
            )
        for bin_index, (prediction_sum, target_sum, count) in enumerate(
            zip(
                self.expected_score_bin_prediction_sums,
                self.expected_score_bin_target_sums,
                self.expected_score_bin_counts,
            )
        ):
            if count == 0:
                continue
            bin_prefix = f'{prefix}/expected_score_calibration/bin_{bin_index:02d}'
            log_scalar(f'{bin_prefix}_predicted', prediction_sum / count, step)
            log_scalar(f'{bin_prefix}_target', target_sum / count, step)
            log_scalar(f'{bin_prefix}_count', count, step)

    def __add__(self, other: ValueMetrics) -> ValueMetrics:
        return ValueMetrics(
            outcome_cross_entropy_sum=self.outcome_cross_entropy_sum + other.outcome_cross_entropy_sum,
            brier_score_sum=self.brier_score_sum + other.brier_score_sum,
            expected_score_mse_sum=self.expected_score_mse_sum + other.expected_score_mse_sum,
            expected_score_mae_sum=self.expected_score_mae_sum + other.expected_score_mae_sum,
            predicted_expected_score_sum=self.predicted_expected_score_sum + other.predicted_expected_score_sum,
            target_expected_score_sum=self.target_expected_score_sum + other.target_expected_score_sum,
            outcome_target_count=self.outcome_target_count + other.outcome_target_count,
            mcts_huber_sum=self.mcts_huber_sum + other.mcts_huber_sum,
            mcts_target_count=self.mcts_target_count + other.mcts_target_count,
            class_probability_sums=tuple(
                first + second for first, second in zip(self.class_probability_sums, other.class_probability_sums)
            ),
            class_correct_counts=tuple(
                first + second for first, second in zip(self.class_correct_counts, other.class_correct_counts)
            ),
            class_target_counts=tuple(
                first + second for first, second in zip(self.class_target_counts, other.class_target_counts)
            ),
            expected_score_bin_prediction_sums=tuple(
                first + second
                for first, second in zip(
                    self.expected_score_bin_prediction_sums,
                    other.expected_score_bin_prediction_sums,
                )
            ),
            expected_score_bin_target_sums=tuple(
                first + second
                for first, second in zip(
                    self.expected_score_bin_target_sums,
                    other.expected_score_bin_target_sums,
                )
            ),
            expected_score_bin_counts=tuple(
                first + second
                for first, second in zip(
                    self.expected_score_bin_counts,
                    other.expected_score_bin_counts,
                )
            ),
        )


@dataclass(frozen=True)
class TrainingStats:
    policy_loss_sum: float
    sample_count: int
    value_metrics: ValueMetrics
    termination_value_metrics: tuple[ValueMetrics, ...]
    value_sum: float
    value_square_sum: float
    gradient_norm_sum: float
    gradient_norm_count: int
    num_batches: int
    outcome_value_loss_weight: float
    mcts_value_loss_weight: float
    policy_loss_weight: float
    value_loss_weight: float

    def __post_init__(self) -> None:
        if len(self.termination_value_metrics) != len(TerminationReason):
            raise ValueError('Training statistics require one value-metric split per termination reason.')

    @property
    def policy_loss(self) -> float:
        return self.policy_loss_sum / self.sample_count if self.sample_count else 0.0

    @property
    def value_loss(self) -> float:
        return self.value_metrics.combined_value_loss(
            self.outcome_value_loss_weight,
            self.mcts_value_loss_weight,
        )

    @property
    def total_loss(self) -> float:
        return self.policy_loss_weight * self.policy_loss + self.value_loss_weight * self.value_loss

    @property
    def value_mean(self) -> float:
        return self.value_sum / self.sample_count if self.sample_count else 0.0

    @property
    def value_std(self) -> float:
        if self.sample_count < 2:
            return 0.0
        centered_square_sum = self.value_square_sum - self.value_sum**2 / self.sample_count
        return sqrt(max(0.0, centered_square_sum / (self.sample_count - 1)))

    @property
    def gradient_norm(self) -> float:
        return self.gradient_norm_sum / self.gradient_norm_count if self.gradient_norm_count else 0.0

    @property
    def excluded_outcome_target_count(self) -> int:
        return self.sample_count - self.value_metrics.outcome_target_count

    def log_to_tensorboard(self, iteration: int, prefix: str) -> None:
        log_scalar(f'{prefix}/policy_loss', self.policy_loss, iteration)
        log_scalar(f'{prefix}/value_loss', self.value_loss, iteration)
        log_scalar(f'{prefix}/total_loss', self.total_loss, iteration)
        log_scalar(f'{prefix}/value_mean', self.value_mean, iteration)
        log_scalar(f'{prefix}/value_std', self.value_std, iteration)
        log_scalar(
            f'{prefix}/outcome_target_excluded_count',
            self.excluded_outcome_target_count,
            iteration,
        )
        self.value_metrics.log_to_tensorboard(f'{prefix}/value', iteration)
        for reason in TerminationReason:
            metrics = self.termination_value_metrics[int(reason)]
            if metrics.outcome_target_count or metrics.mcts_target_count:
                metrics.log_to_tensorboard(
                    f'{prefix}/value_by_termination/{reason.name.lower()}',
                    iteration,
                )
        if self.gradient_norm > 0:
            log_scalar(f'{prefix}/gradient_norm', self.gradient_norm, iteration)

    def __repr__(self) -> str:
        return (
            f'Policy Loss: {self.policy_loss:.4f}, Value Loss: {self.value_loss:.4f}, '
            f'WDL CE: {self.value_metrics.outcome_cross_entropy:.4f}, '
            f'MCTS Huber: {self.value_metrics.mcts_huber:.4f}, '
            f'Total Loss: {self.total_loss:.4f}, Value Mean: {self.value_mean:.4f}, '
            f'Value Std: {self.value_std:.4f}, Gradient Norm: {self.gradient_norm:.4f}, '
            f'Num Batches: {self.num_batches}, Samples: {self.sample_count}'
        )

    @staticmethod
    def combine(stats_list: list[TrainingStats]) -> TrainingStats:
        if not stats_list:
            raise ValueError('At least one training-statistics value is required.')
        outcome_weight = stats_list[0].outcome_value_loss_weight
        mcts_weight = stats_list[0].mcts_value_loss_weight
        policy_weight = stats_list[0].policy_loss_weight
        value_weight = stats_list[0].value_loss_weight
        if any(
            stats.outcome_value_loss_weight != outcome_weight
            or stats.mcts_value_loss_weight != mcts_weight
            or stats.policy_loss_weight != policy_weight
            or stats.value_loss_weight != value_weight
            for stats in stats_list
        ):
            raise ValueError('Cannot combine training statistics with different loss weights.')
        return TrainingStats(
            policy_loss_sum=sum(stats.policy_loss_sum for stats in stats_list),
            sample_count=sum(stats.sample_count for stats in stats_list),
            value_metrics=sum(
                (stats.value_metrics for stats in stats_list),
                start=ValueMetrics(),
            ),
            termination_value_metrics=tuple(
                sum(
                    (stats.termination_value_metrics[int(reason)] for stats in stats_list),
                    start=ValueMetrics(),
                )
                for reason in TerminationReason
            ),
            value_sum=sum(stats.value_sum for stats in stats_list),
            value_square_sum=sum(stats.value_square_sum for stats in stats_list),
            gradient_norm_sum=sum(stats.gradient_norm_sum for stats in stats_list),
            gradient_norm_count=sum(stats.gradient_norm_count for stats in stats_list),
            num_batches=sum(stats.num_batches for stats in stats_list),
            outcome_value_loss_weight=outcome_weight,
            mcts_value_loss_weight=mcts_weight,
            policy_loss_weight=policy_weight,
            value_loss_weight=value_weight,
        )
