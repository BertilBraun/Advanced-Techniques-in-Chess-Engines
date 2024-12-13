import time
from pathlib import Path

from AIZeroConnect4Bot.src.util import random_id


class ClusterManager:
    def __init__(self, num_nodes: int, communication_dir: str = 'communication') -> None:
        self.communication_dir = Path(communication_dir)
        self.communication_dir.mkdir(exist_ok=True)
        self.size = num_nodes
        self.rank = -1

    @property
    def is_root_node(self) -> bool:
        assert self.rank != -1, 'ClusterManager not initialized'
        return self.rank == 0

    def initialize(self) -> None:
        # Clean up previous communication files
        for f in self.communication_dir.iterdir():
            f.unlink(missing_ok=True)

        my_id = random_id()

        print('Node initialized')

        log_file = self.communication_dir / f'initialize_{my_id}.txt'
        while True:
            # Create a file to signal this node's readiness
            open(log_file, 'w').close()

            # Count how many have initialized
            initialized_nodes = list(self.communication_dir.iterdir())
            if len(initialized_nodes) == self.size:
                break

            print(f'Waiting for {self.size - len(initialized_nodes)} nodes to initialize')
            time.sleep(1)

        print('All nodes initialized')

        # Determine the root node (lowest ID)
        all_ids = [f.stem.replace('initialize_', '') for f in initialized_nodes]
        self.rank = all_ids.index(my_id)
        self.size = len(all_ids)

        if self.is_root_node:
            # Clean up the initialization files
            for f in initialized_nodes:
                f.unlink(missing_ok=True)

    def barrier(self, name: str) -> None:
        # Create a barrier file for this node
        log_file = self.communication_dir / f'{self.rank}{"_" + name if name else ""}.txt'
        open(log_file, 'w').close()

        print(f'Node {self.rank} reached the barrier {name}')
        while True:
            written_files = len([f for f in self.communication_dir.iterdir() if f.stem.endswith(name)])
            time.sleep(1)

            if written_files == self.size:
                break

        # remove the barrier file after all have reached
        if self.is_root_node:
            for f in self.communication_dir.iterdir():
                if f.stem.endswith(name):
                    f.unlink(missing_ok=True)

        print(f'All nodes have reached the barrier {name}')