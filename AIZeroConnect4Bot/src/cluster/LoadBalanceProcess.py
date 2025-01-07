import time

from src.cluster.LoadBalancer import LoadBalancer
from src.util.exceptions import log_exceptions
from src.util.PipeConnection import PipeConnection


def run_load_balancer_process(
    load_balancer_input_pipes: list[PipeConnection], load_balancer_output_pipes: list[PipeConnection]
):
    all_readable_and_writable = all(
        pipe.readable and pipe.writable for pipe in load_balancer_input_pipes + load_balancer_output_pipes
    )
    assert all_readable_and_writable, 'All pipes must be readable and writable.'

    load_balancer = LoadBalancerProcess(load_balancer_input_pipes, load_balancer_output_pipes)
    with log_exceptions('Load balancer'):
        load_balancer.run()


class LoadBalancerProcess:
    def __init__(self, input_pipes: list[PipeConnection], output_pipes: list[PipeConnection]) -> None:
        self.load_balancer = LoadBalancer(output_pipes)
        self.input_pipes = input_pipes

    def run(self):
        while True:
            for pipe in self.input_pipes:
                while pipe.poll():
                    message = pipe.recv_bytes()
                    self.load_balancer.send_request(message, pipe)

            for response, pipe in self.load_balancer.recieve_responses():
                pipe.send_bytes(response)

            time.sleep(0.00001)
