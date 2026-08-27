from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from tools.distill_train_student import (
    Arguments,
    AttentionBiasKind,
    LearningRateSchedule,
    NetworkKind,
    OptimizerKind,
    PolicyHeadKind,
)

# The probe-settled supervised protocol, shared by every cell so that only the architecture differs.
PROTOCOL_STEPS = 80_000
PROTOCOL_BATCH_SIZE = 1024
PROTOCOL_PEAK_LEARNING_RATE = 0.002
PROTOCOL_WARMUP_STEPS = 200
PROTOCOL_ANNEAL_FRACTION = 0.2
PROTOCOL_MAX_GRAD_NORM = 0.5
PROTOCOL_CHECKPOINT_EVERY = 10_000
PROTOCOL_EVALUATE_EVERY = 2_000
PROTOCOL_RANDOM_SEED = 20260827

# The secondary experiment runs on the convolutional cell, at production's own peak rate.
SCHEDULE_PEAK_LEARNING_RATE = 0.005
# AlphaZero's 0.2 was set at batch 4096; linear scaling puts batch 1024 at a quarter of it.
SGD_PEAK_LEARNING_RATE = 0.05

BASE = Arguments(
    dataset=Path('/workspace/distill/attention-6m.bin'),
    output_run_state=Path('/workspace/attention-cells'),
    network_kind=NetworkKind.CONVOLUTIONAL,
    layers=12,
    hidden_size=128,
    heads=8,
    feedforward=256,
    policy_head_kind=PolicyHeadKind.DENSE,
    policy_key_size=128,
    attention_bias_kind=AttentionBiasKind.DISABLED,
    smolgen_compressed_size=8,
    smolgen_hidden_size=32,
    smolgen_generated_size=32,
    optimizer_kind=OptimizerKind.ADAMW,
    floor_fraction=0.1,
    policy_bottleneck_rank=0,
    batch_size=PROTOCOL_BATCH_SIZE,
    steps=PROTOCOL_STEPS,
    learning_rate=PROTOCOL_PEAK_LEARNING_RATE,
    learning_rate_schedule=LearningRateSchedule.PLATEAU,
    anneal_fraction=PROTOCOL_ANNEAL_FRACTION,
    warmup_steps=PROTOCOL_WARMUP_STEPS,
    max_grad_norm=PROTOCOL_MAX_GRAD_NORM,
    holdout_fraction=0.02,
    training_fraction=1.0,
    evaluate_every=PROTOCOL_EVALUATE_EVERY,
    checkpoint_every=PROTOCOL_CHECKPOINT_EVERY,
    distil_auxiliary_heads=(),
    device_id=0,
    random_seed=PROTOCOL_RANDOM_SEED,
    generation=322,
)

ATTENTION = replace(BASE, network_kind=NetworkKind.ATTENTION, hidden_size=176, heads=11, feedforward=352)


@dataclass(frozen=True)
class Cell:
    name: str
    question: str
    arguments: Arguments


ARCHITECTURE_CELLS = (
    Cell('cnn-A', 'the convolutional baseline', BASE),
    Cell(
        'attn-A',
        'attention as it stands, with the generation-0 bootstrap bug fixed',
        replace(ATTENTION, layers=14),
    ),
    Cell(
        'attn-B',
        'attention with a from-to policy head and the saved parameters back in the trunk',
        replace(ATTENTION, layers=15, policy_head_kind=PolicyHeadKind.FROM_TO_ATTENTION),
    ),
    Cell(
        'attn-C',
        'attn-B plus a generated attention bias',
        replace(
            ATTENTION,
            layers=13,
            policy_head_kind=PolicyHeadKind.FROM_TO_ATTENTION,
            attention_bias_kind=AttentionBiasKind.SMOLGEN,
        ),
    ),
)

SCHEDULE_CELLS = (
    Cell(
        'lr-production-flat',
        'the near-flat schedule production v9 runs, 1.67x total decay',
        replace(
            BASE,
            learning_rate=SCHEDULE_PEAK_LEARNING_RATE,
            learning_rate_schedule=LearningRateSchedule.PRODUCTION_FLAT,
        ),
    ),
    Cell(
        'lr-staged-decay',
        'the proposed v10 staged drops, 10x total decay',
        replace(
            BASE,
            learning_rate=SCHEDULE_PEAK_LEARNING_RATE,
            learning_rate_schedule=LearningRateSchedule.STAGED_DECAY,
        ),
    ),
    Cell(
        'lr-cosine-floor',
        'a cosine anneal to the same 10x floor',
        replace(
            BASE,
            learning_rate=SCHEDULE_PEAK_LEARNING_RATE,
            learning_rate_schedule=LearningRateSchedule.COSINE_FLOOR,
        ),
    ),
    Cell(
        'lr-sgd-alphazero',
        "SGD with momentum on AlphaZero's thousandfold step shape",
        replace(
            BASE,
            learning_rate=SGD_PEAK_LEARNING_RATE,
            learning_rate_schedule=LearningRateSchedule.STAGED_DECAY,
            optimizer_kind=OptimizerKind.SGD_MOMENTUM,
        ),
    ),
)

# attn-B beats both cnn-A and attn-A from early on, so the question stops being "is attention weaker"
# and becomes "is the win the trunk or the head". cnn-from-to is that control: the same from-to head on
# a convolutional trunk widened to 136 channels so the total parameter count still matches cnn-A.
SUPPLEMENTARY_CELLS = (
    Cell(
        'cnn-from-to',
        'the from-to policy head on a convolutional trunk, at cnn-A parameters',
        replace(BASE, hidden_size=136, policy_head_kind=PolicyHeadKind.FROM_TO_ATTENTION),
    ),
)

# cnn-from-to changed two things at once: the head, and a trunk widened from 128 to 136 channels to spend
# the head's parameter saving. These two complete a 2x2 over (trunk width) x (head), so the head's own
# contribution can be read off at a fixed trunk and the widening's at a fixed head. They also settle the
# throughput question: 90% of cnn-from-to's extra multiply-accumulates are the widening, not the head.
HEAD_CONTROL_CELLS = (
    Cell(
        'cnn-from-to-narrow',
        'the from-to head at the production trunk width, isolating the head from the widening',
        replace(BASE, policy_head_kind=PolicyHeadKind.FROM_TO_ATTENTION),
    ),
    Cell(
        'cnn-dense-wide',
        'the widened trunk keeping the dense head, isolating the widening from the head',
        replace(BASE, hidden_size=136),
    ),
)

ALL_CELLS = ARCHITECTURE_CELLS + SCHEDULE_CELLS + SUPPLEMENTARY_CELLS + HEAD_CONTROL_CELLS


def cell_by_name(name: str) -> Cell:
    for cell in ALL_CELLS:
        if cell.name == name:
            return cell
    raise ValueError(f'No cell is named {name!r}; the cells are {tuple(cell.name for cell in ALL_CELLS)}.')


def command_line(cell: Cell, output_root: Path, python: str = 'python') -> tuple[str, ...]:
    arguments = cell.arguments
    command = [
        python,
        '-m',
        'tools.distill_train_student',
        '--dataset',
        str(arguments.dataset),
        '--output-run-state',
        str(output_root / cell.name),
        '--network-kind',
        arguments.network_kind.value,
        '--layers',
        str(arguments.layers),
        '--hidden-size',
        str(arguments.hidden_size),
        '--heads',
        str(arguments.heads),
        '--feedforward',
        str(arguments.feedforward),
        '--policy-head-kind',
        arguments.policy_head_kind.value,
        '--policy-key-size',
        str(arguments.policy_key_size),
        '--attention-bias-kind',
        arguments.attention_bias_kind.value,
        '--smolgen-compressed-size',
        str(arguments.smolgen_compressed_size),
        '--smolgen-hidden-size',
        str(arguments.smolgen_hidden_size),
        '--smolgen-generated-size',
        str(arguments.smolgen_generated_size),
        '--optimizer',
        arguments.optimizer_kind.value,
        '--floor-fraction',
        str(arguments.floor_fraction),
        '--policy-bottleneck-rank',
        str(arguments.policy_bottleneck_rank),
        '--batch-size',
        str(arguments.batch_size),
        '--steps',
        str(arguments.steps),
        '--learning-rate',
        str(arguments.learning_rate),
        '--learning-rate-schedule',
        arguments.learning_rate_schedule.value,
        '--anneal-fraction',
        str(arguments.anneal_fraction),
        '--warmup-steps',
        str(arguments.warmup_steps),
        '--max-grad-norm',
        str(arguments.max_grad_norm),
        '--evaluate-every',
        str(arguments.evaluate_every),
        '--checkpoint-every',
        str(arguments.checkpoint_every),
        '--device-id',
        str(arguments.device_id),
        '--random-seed',
        str(arguments.random_seed),
        '--generation',
        str(arguments.generation),
    ]
    return tuple(command)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Print the command line for one attention-viability cell.')
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--python', default='python')
    parser.add_argument('--steps', default=0, type=int, help='Override the step count, for a short probe.')
    namespace = parser.parse_args()
    cell = cell_by_name(namespace.cell)
    if namespace.steps:
        cell = replace(cell, arguments=replace(cell.arguments, steps=namespace.steps))
    print(' '.join(command_line(cell, namespace.output_root, namespace.python)))


if __name__ == '__main__':
    main()
