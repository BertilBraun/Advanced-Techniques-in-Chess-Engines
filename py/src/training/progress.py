from pydantic import Field

from src.util.frozen_model import FrozenModel


class TrainingProgress(FrozenModel):
    completed_optimizer_steps: int = Field(ge=0)
