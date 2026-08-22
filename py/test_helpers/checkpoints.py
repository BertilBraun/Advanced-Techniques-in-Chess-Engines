from __future__ import annotations

import hashlib
from pathlib import Path

from src.games.representation import NetworkDimensions
from src.training.checkpoint import CheckpointReference
from src.training.checkpoint.contracts import CheckpointManifest
from src.training.checkpoint.paths import checkpoint_manifest_path
from src.training.network import (
    DisabledResidualContext,
    GoPointPassPolicyHeadConfiguration,
    NetworkDefinition,
    NetworkParams,
)
from src.util.atomic_file import write_text_atomically


def checkpoint_reference(
    directory: Path = Path('.'),
    generation: int = 1,
    *,
    write_inference_model: bool = False,
) -> CheckpointReference:
    inference_model_path = directory / f'model_{generation}.jit.pt'
    if write_inference_model:
        directory.mkdir(parents=True, exist_ok=True)
        inference_model_path.write_bytes(f'model {generation}'.encode('ascii'))
        digest = hashlib.sha256(inference_model_path.read_bytes()).hexdigest()
    else:
        digest = f'{generation:064x}'
    return CheckpointReference(
        generation=generation,
        manifest_path=checkpoint_manifest_path(generation, directory),
        model_path=directory / f'model_{generation}.pt',
        optimizer_path=directory / f'optimizer_{generation}.pt',
        inference_model_path=inference_model_path,
        inference_model_sha256=digest,
    )


def _minimal_network_definition() -> NetworkDefinition:
    return NetworkDefinition(
        architecture=NetworkParams(
            num_layers=1,
            hidden_size=8,
            residual_context=DisabledResidualContext(),
            policy_head=GoPointPassPolicyHeadConfiguration(),
            num_value_channels=1,
            value_fc_size=8,
        ),
        dimensions=NetworkDimensions(channels=3, rows=7, columns=7, actions=50, outcomes=3),
        auxiliary_heads=(),
    )


def materialized_checkpoint(directory: Path, generation: int) -> CheckpointReference:
    directory.mkdir(parents=True, exist_ok=True)
    reference = checkpoint_reference(directory, generation)
    payloads = {
        reference.model_path: f'model {generation}'.encode('ascii'),
        reference.optimizer_path: f'optimizer {generation}'.encode('ascii'),
        reference.inference_model_path: f'inference {generation}'.encode('ascii'),
    }
    for path, payload in payloads.items():
        path.write_bytes(payload)
    manifest = CheckpointManifest(
        generation=generation,
        network=_minimal_network_definition(),
        model_path=reference.model_path.name,
        model_sha256=hashlib.sha256(payloads[reference.model_path]).hexdigest(),
        optimizer_path=reference.optimizer_path.name,
        optimizer_sha256=hashlib.sha256(payloads[reference.optimizer_path]).hexdigest(),
        inference_model_path=reference.inference_model_path.name,
        inference_model_sha256=hashlib.sha256(payloads[reference.inference_model_path]).hexdigest(),
    )
    write_text_atomically(reference.manifest_path, manifest.model_dump_json(indent=2) + '\n')
    return reference.model_copy(update={'inference_model_sha256': manifest.inference_model_sha256})
