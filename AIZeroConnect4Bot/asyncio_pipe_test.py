import asyncio
import multiprocessing

from asyncio_pipe import Connection


def send(connection, obj):
    connection.send(obj)


def recv(connection):
    return connection.recv()


def send_bytes(connection, buf, offset=0, size=None):
    connection.send_bytes(buf, offset, size)


def recv_bytes(connection, maxlength=None):
    return connection.recv_bytes(maxlength)


async def verify_duplex_connection(async_connection, connection):
    connection.send([1, 2, 3])
    assert await async_connection.recv() == [1, 2, 3]
    async_connection.send([4, 5, 6])
    assert connection.recv() == [4, 5, 6]

    # test bytes
    connection.send_bytes(b'hello')
    assert await async_connection.recv_bytes() == b'hello'
    async_connection.send_bytes(b'world')
    assert connection.recv_bytes() == b'world'

    # test on other process
    p = multiprocessing.Process(target=send, args=(connection, [1, 2, 3]))
    p.start()

    assert await async_connection.recv() == [1, 2, 3]

    p = multiprocessing.Process(target=send_bytes, args=(connection, b'hello'))
    p.start()

    assert await async_connection.recv_bytes() == b'hello'

    p = multiprocessing.Process(target=send, args=(async_connection, [4, 5, 6]))
    p.start()

    assert connection.recv() == [4, 5, 6]

    p = multiprocessing.Process(target=send_bytes, args=(async_connection, b'world'))
    p.start()

    assert connection.recv_bytes() == b'world'

    connection.close()
    async_connection.close()


async def test_async_connection():
    c1, c2 = multiprocessing.Pipe()
    await verify_duplex_connection(Connection(c1), c2)


asyncio.run(test_async_connection())
