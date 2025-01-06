from multiprocessing.connection import PipeConnection

from src.settings import TRAINING_ARGS
from src.util.log import log
from src.alpha_zero.SelfPlay import SelfPlay
from src.cluster.InferenceClient import InferenceClient
from src.util.exceptions import log_exceptions
from src.alpha_zero.train.TrainingArgs import TrainingArgs


def run_self_play_process(commander_pipe: PipeConnection, inference_server_pipe: PipeConnection):
    assert commander_pipe.readable and not commander_pipe.writable, 'Commander pipe must be readable and not writable.'

    client = InferenceClient(inference_server_pipe)
    self_play_process = SelfPlayProcess(client, TRAINING_ARGS, commander_pipe)
    with log_exceptions('Self play process'):
        self_play_process.run()


class SelfPlayProcess:
    def __init__(self, client: InferenceClient, args: TrainingArgs, commander_pipe: PipeConnection) -> None:
        self.args = args
        self.self_play = SelfPlay(client, args.self_play)
        self.commander_pipe = commander_pipe

    def run(self):
        current_iteration = 0
        running = False

        while True:
            if running:
                with log_exceptions('Self play'):
                    dataset = self.self_play.self_play(current_iteration)

                    log(f'Collected {len(dataset)} self-play memories.')
                    dataset.save(self.args.save_path, current_iteration)

            if self.commander_pipe.poll():
                message = self.commander_pipe.recv()
                assert isinstance(message, str), f'Expected message to be a string, got {message}'
                if message == 'STOP':
                    break
                elif message.startswith('START AT ITERATION:'):
                    current_iteration = int(message.split(':')[-1])
                    running = True

        log('Self play process stopped.')
