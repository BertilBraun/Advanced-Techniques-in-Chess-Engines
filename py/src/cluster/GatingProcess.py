from src.eval.ModelEvaluationCpp import ModelEvaluation
from src.train.TrainingArgs import TrainingArgs
from src.util.communication import Communication
from src.util.exceptions import log_exceptions
from src.util.log import log
from src.util.save_paths import model_save_path
from src.util.tensorboard import TensorboardWriter, log_scalar, log_scalars


class GatingProcess:
    def __init__(self, args: TrainingArgs, run: int, trainer_device_id: int):
        self.run_id = run
        self.args = args
        self.trainer_device_id = trainer_device_id

        self.communication_folder = f'communication/{self.run_id}'
        self.communication = Communication(self.communication_folder)

    def run(self, iteration: int, current_best_iteration: int) -> int:
        if self.args.gating is None:
            # ignore gating, always update the model as quickly as possible
            current_best_iteration = iteration + 1
            self.communication.boardcast(f'LOAD MODEL: {current_best_iteration}')
            log(f'Gating evaluation skipped at iteration {iteration}. Using model {current_best_iteration}.')
            return current_best_iteration

        log(f'Running gating evaluation at iteration {iteration}.')

        with TensorboardWriter(self.run_id, 'gating', postfix_pid=False), log_exceptions('Gating evaluation'):
            gating_evaluation = ModelEvaluation(
                iteration + 1,
                self.args,
                device_id=self.trainer_device_id,
                num_games=self.args.gating.num_games,
                num_searches_per_turn=self.args.gating.num_searches_per_turn,
            )
            results = gating_evaluation.play_two_models_search(
                model_save_path(current_best_iteration, self.args.save_path)
            )

            log_scalars(
                'gating/gating',
                {
                    'wins': results.wins,
                    'losses': results.losses,
                    'draws': results.draws,
                },
                iteration,
            )

            if self.args.gating.ignore_draws:
                result_score = (
                    results.wins / (results.wins + results.losses) if results.wins + results.losses > 0 else 0.0
                )
            else:
                result_score = (results.wins + results.draws * 0.5) / gating_evaluation.num_games
            log(f'Gating evaluation at iteration {iteration} resulted in {result_score} score ({results}).')
            if result_score > self.args.gating.gating_threshold:
                log(f'Gating evaluation passed at iteration {iteration}.')
                current_best_iteration = iteration + 1
                self.communication.boardcast(f'LOAD MODEL: {current_best_iteration}')
            else:
                log(
                    f'Gating evaluation failed at iteration {iteration}.'
                    f'Keeping current best model {current_best_iteration}.'
                )

            log_scalar('gating/current_best_iteration', current_best_iteration, iteration)
            log_scalar('gating/gating_score', result_score, iteration)

        return current_best_iteration
