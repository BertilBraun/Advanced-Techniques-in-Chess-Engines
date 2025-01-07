import asyncio
import multiprocessing

from asyncio_pipe import Connection

from src.cluster.InferenceClient import InferenceClient
from src.cluster.InferenceServerProcess import run_inference_server

from src.settings import CurrentBoard


pipe1, pipe2 = multiprocessing.Pipe()
comm1, comm2 = multiprocessing.Pipe()

p1 = multiprocessing.Process(target=run_inference_server, args=(pipe1, comm1, 0))
p1.start()

comm2.send('START AT ITERATION: 0')
client = InferenceClient(pipe2)


async def main():
    await client.inference([CurrentBoard()])


asyncio.run(main())
p1.join()
