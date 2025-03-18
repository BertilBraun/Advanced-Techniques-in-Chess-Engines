from __future__ import annotations
import numpy as np
import zmq
import pickle

from src.cluster.InferenceCache import InferenceCache
from src.settings import CurrentBoard, CurrentGame

from src.Encoding import MoveScore, encode_board_state
from src.util.ZobristHasherNumpy import ZobristHasherNumpy


class InferenceClient:
    """The Inference Client is responsible for batching and caching inference requests. It uses a model to directly infer the policy and value for a given board state on the provided device."""

    def __init__(self, server_address: str):
        self.context = zmq.Context.instance()
        # Use a DEALER socket so we can send many messages concurrently.
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.connect(server_address)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        # Simple in-memory cache for inference results.
        self.local_cache = InferenceCache()
        # For generating unique correlation IDs.
        self._corr_id_counter = 0

        channels, rows, cols = CurrentGame.representation_shape
        self.hasher = ZobristHasherNumpy(channels, rows, cols)

    def _next_corr_id(self):
        cid = self._corr_id_counter
        self._corr_id_counter += 1
        return cid

    def inference_batch(self, inputs: list[CurrentBoard]) -> list[tuple[list[tuple[int, float]], float]]:
        """
        Accepts a list of inputs that are not currently cached.
        For each input not in cache, a message is sent to the server,
        the responses are received and the cache is updated.
        Returns a list of inference results in the same order as 'inputs'.
        """

        encoded_boards = [CurrentGame.get_canonical_board(board) for board in inputs]
        board_hashes = self.hasher.zobrist_hash_boards(np.array(encoded_boards))

        # Prepare lists for inputs that require an inference request.
        request_hashes, to_request = self.local_cache.filter(board_hashes, encoded_boards)

        if to_request:
            # Create a mapping from correlation ID to the hash in results.
            corr_map: dict[int, int] = {}
            request: list[tuple[int, np.ndarray]] = []

            for hash, board in zip(request_hashes, to_request):
                corr_id = self._next_corr_id()
                corr_map[corr_id] = hash
                request.append((corr_id, encode_board_state(board)))

            # Send the message over the DEALER socket.
            self.socket.send(pickle.dumps(request))

            pending = set(corr_map.keys())
            while pending:
                socks = dict(self.poller.poll(timeout=1000))
                if self.socket in socks:
                    payload = self.socket.recv()
                    try:
                        responses = pickle.loads(payload)
                        for corr_id, result in responses:
                            hash = corr_map[corr_id]
                            moves_np, value = result
                            self.local_cache.add(hash, moves_np, value)
                            pending.remove(corr_id)

                        assert not pending, 'Received a response for an unknown correlation ID.'
                    except Exception as e:
                        print('Error decoding response:', e)

        results: list[tuple[list[MoveScore], float]] = []
        for hash in board_hashes:
            moves, value = self.local_cache.get(hash)
            results.append((moves, value))

        return results
