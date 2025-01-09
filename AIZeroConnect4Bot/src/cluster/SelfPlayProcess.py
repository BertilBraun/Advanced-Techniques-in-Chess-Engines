import asyncio
import time
from torch.multiprocessing import Queue

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.settings import TRAINING_ARGS
from src.util.log import log
from src.alpha_zero.SelfPlay import SelfPlay
from src.cluster.InferenceClient import InferenceClient
from src.util.exceptions import log_exceptions
from src.alpha_zero.train.TrainingArgs import TrainingArgs
from src.util.PipeConnection import PipeConnection


def run_self_play_process(commander_pipe: PipeConnection, inference_queue: Queue, result_queue: Queue, global_id: int):
    assert commander_pipe.readable and not commander_pipe.writable, 'Commander pipe must be readable and not writable.'

    client = InferenceClient(inference_queue, result_queue, global_id)
    self_play_process = SelfPlayProcess(client, TRAINING_ARGS, commander_pipe)
    with log_exceptions('Self play process crashed.'):
        asyncio.run(self_play_process.run())


class SelfPlayProcess:
    def __init__(self, client: InferenceClient, args: TrainingArgs, commander_pipe: PipeConnection) -> None:
        self.args = args
        self.self_play = SelfPlay(client, args.self_play)
        self.commander_pipe = commander_pipe
        self.start_generating_samples = time.time()

    async def run(self):
        current_iteration = 0
        running = False

        while True:
            if running:
                await self.self_play.self_play()

                if len(self.self_play.dataset) >= 1000:
                    self._save_dataset(current_iteration)

            if self.commander_pipe.poll():
                message = self.commander_pipe.recv()
                assert isinstance(message, str), f'Expected message to be a string, got {message}'
                if message == 'STOP':
                    break
                elif message.startswith('START AT ITERATION:'):
                    current_iteration = int(message.split(':')[-1])
                    running = True
                    self._save_dataset(current_iteration)
                    self.self_play.client.reset_cache()
                    self.self_play.update_iteration(current_iteration)

        log('Self play process stopped.')

    def _save_dataset(self, iteration: int):
        if not len(self.self_play.dataset):
            return

        time_to_generate = time.time() - self.start_generating_samples
        log(f'Generating {len(self.self_play.dataset)} samples took {time_to_generate}sec')
        self.self_play.dataset.save(self.args.save_path, iteration)
        self.self_play.dataset = SelfPlayDataset()
        self.start_generating_samples = time.time()
