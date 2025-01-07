import asyncio
import multiprocessing

from asyncio_pipe import Connection

from src.cluster.InferenceClient import InferenceClient
from src.cluster.InferenceServerProcess import run_inference_server

from src.settings import CurrentBoard


pipe1, pipe2 = multiprocessing.Pipe()
comm1, comm2 = multiprocessing.Pipe()

print('Starting inference server')
p1 = multiprocessing.Process(target=run_inference_server, args=(pipe1, comm1, 0))
p1.start()

print('Sending start message')

comm2.send('START AT ITERATION: 0')

print('Creating client')

client = InferenceClient(pipe2)

print('Running inference')


async def main():
    print('Inference result:')
    res = await client.inference([CurrentBoard()])
    print(res)
    print('Inference done')


asyncio.run(main())

print('Sending stop message')
p1.join()
