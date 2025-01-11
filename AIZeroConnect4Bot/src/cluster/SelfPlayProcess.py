import time
import asyncio

from src.alpha_zero.SelfPlayDataset import SelfPlayDataset
from src.util.log import log
from src.alpha_zero.SelfPlay import SelfPlay
from src.cluster.InferenceClient import InferenceClient
from src.util.exceptions import log_exceptions
from src.alpha_zero.train.TrainingArgs import SelfPlayParams, TrainingArgs
from src.util.PipeConnection import PipeConnection


def run_self_play_process(args: TrainingArgs, commander_pipe: PipeConnection, device_id: int):
    assert commander_pipe.readable and not commander_pipe.writable, 'Commander pipe must be readable and not writable.'

    client = InferenceClient(device_id, args.network, args.inference)
    self_play_process = SelfPlayProcess(client, args.self_play, args.save_path, commander_pipe)
    with log_exceptions(f'Self play process {device_id} crashed.'):
        asyncio.run(self_play_process.run())


class SelfPlayProcess:
    def __init__(
        self, client: InferenceClient, args: SelfPlayParams, save_path: str, commander_pipe: PipeConnection
    ) -> None:
        self.save_path = save_path
        self.args = args
        self.self_play = SelfPlay(client, args)
        self.commander_pipe = commander_pipe
        self.start_time_of_generating_samples = time.time()

    async def run(self):
        current_iteration = 0
        running = False

        while True:
            if running:
                await self.self_play.self_play()

                if len(self.self_play.dataset) >= self.args.num_samples_after_which_to_write:
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
                    self.self_play.update_iteration(current_iteration)

        log('Self play process stopped.')

    def _save_dataset(self, iteration: int):
        if not len(self.self_play.dataset):
            return

        time_to_generate = time.time() - self.start_time_of_generating_samples
        log(f'Generating {len(self.self_play.dataset)} samples took {time_to_generate:.2f}sec')
        self.self_play.dataset.save(self.save_path, iteration)
        self.self_play.dataset = SelfPlayDataset()
        self.start_time_of_generating_samples = time.time()
