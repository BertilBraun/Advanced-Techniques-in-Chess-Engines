from src.self_play.SelfPlayDataset import SelfPlayDataset
from src.settings import TensorboardWriter
from src.self_play.SelfPlay import SelfPlay
from src.cluster.InferenceClient import InferenceClient
from src.util.exceptions import log_exceptions
from src.train.TrainingArgs import SelfPlayParams, TrainingArgs
from src.util.profiler import start_cpu_usage_logger
from src.util.save_paths import get_latest_model_iteration, model_save_path


def run_self_play_process(run: int, args: TrainingArgs, server_address: str):
    device_id = server_address.split('/').pop()
    if device_id == 0:
        start_cpu_usage_logger(run, 'self_play_cpu_usage')

    client = InferenceClient(server_address)
    self_play_process = SelfPlayProcess(client, args.self_play, args.save_path)
    with log_exceptions(f'Self play process {server_address} crashed.'), TensorboardWriter(run, 'self_play'):
        self_play_process.run()


class SelfPlayProcess:
    """This class provides functionality to run the self play process. It runs self play games and saves the dataset to disk. It listens to the commander for messages to start and stop the self play process."""

    def __init__(self, client: InferenceClient, args: SelfPlayParams, save_path: str) -> None:
        self.save_path = save_path
        self.args = args
        self.self_play = SelfPlay(client, args)

    def run(self):
        current_iteration = get_latest_model_iteration(self.save_path)

        while True:
            self.self_play.self_play()

            if self.self_play.dataset.stats.num_games >= self.args.num_games_after_which_to_write:
                self._save_dataset(current_iteration)

            if model_save_path(current_iteration + 1, self.save_path).exists():
                self._save_dataset(current_iteration)
                self.self_play.update_iteration(current_iteration + 1)
                current_iteration += 1

    def _save_dataset(self, iteration: int):
        if not len(self.self_play.dataset):
            return

        self.self_play.dataset.save(self.save_path, iteration)
        self.self_play.dataset = SelfPlayDataset()
