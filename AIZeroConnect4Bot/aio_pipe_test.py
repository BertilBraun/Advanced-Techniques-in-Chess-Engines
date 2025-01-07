from multiprocessing import Process
import asyncio
from aiopipe import aioduplex


async def main():
    mainpipe, chpipe = aioduplex()
    with chpipe.detach() as chpipe:
        proc = Process(target=childproc, args=(chpipe,))
        proc.start()
    # The second pipe is now available in the child process
    # and detached from the parent process.
    async with mainpipe.open() as (rx, tx):
        req = await rx.read(5)
        tx.write(req + b' world\n')
        msg = await rx.readline()
    proc.join()
    return msg


def childproc(pipe):
    asyncio.run(childtask(pipe))


async def childtask(pipe):
    async with pipe.open() as (rx, tx):
        tx.write(b'hello')
        rep = await rx.readline()
        tx.write(rep.upper())


asyncio.run(main())
