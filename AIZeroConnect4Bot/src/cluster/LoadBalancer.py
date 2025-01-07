from typing import Generator
from multiprocessing.connection import _ConnectionBase


class LoadBalancer:
    def __init__(self, server_pipes: list[_ConnectionBase]) -> None:
        self.inference_servers: list[_ConnectionBase] = server_pipes
        self.server_loads = [0] * len(server_pipes)  # Tracks number of pending requests per server
        # Mapping from server to clients and batch sizes
        self.server_to_clients: list[list[_ConnectionBase]] = [[] for _ in range(len(server_pipes))]

    def get_least_loaded_server(self) -> int:
        """Finds the server with the least number of pending requests."""
        return self.server_loads.index(min(self.server_loads))

    def send_request(self, message: bytes, client_conn: _ConnectionBase) -> None:
        """
        Sends an inference request to the least loaded server and maps the client connection.

        Args:
            encoded_board (ndarray): The encoded board state to be inferred.
            client_conn (_ConnectionBase): The Pipe connection to the client.
        """
        server_idx = self.get_least_loaded_server()
        self.inference_servers[server_idx].send_bytes(message)
        self.server_loads[server_idx] += 1
        self.server_to_clients[server_idx].append(client_conn)

    def recieve_responses(self) -> Generator[tuple[bytes, _ConnectionBase], None, None]:
        """
        Generator that yields responses from servers and retrieves the corresponding client connections.

        Yields:
            tuple: A tuple containing (policy, value) and the client connection.
        """
        for server_idx, server_conn in enumerate(self.inference_servers):
            while server_conn.poll():
                response = server_conn.recv_bytes()
                self.server_loads[server_idx] -= 1
                client_conn = self.server_to_clients[server_idx].pop(0)
                yield response, client_conn
