"""Black-box Unix/TCP framing and documented CLI contract checks."""
import os
from pathlib import Path
import socket
import struct
import subprocess
import tempfile
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
TOOL = os.environ.get("NEOGPU_TOOL", "./neogpu_tool")
HEADER = struct.Struct("<IHHII")
REQUEST = struct.Struct("<BBHI")
RESPONSE = struct.Struct("<III")


def exact(connection, count):
    data = bytearray()
    while len(data) < count:
        part = connection.recv(count - len(data))
        assert part, "truncated response"
        data.extend(part)
    return data


def request(connection, op, cid, payload=b""):
    body = REQUEST.pack(op, 0, 0, len(payload)) + payload
    connection.sendall(HEADER.pack(0x4E474950, 1, 1, len(body), cid) + body)
    magic, version, kind, size, returned_cid = HEADER.unpack(exact(connection, HEADER.size))
    assert (magic, version, kind, returned_cid) == (0x4E474950, 1, 2, cid)
    status, result_op, length = RESPONSE.unpack(exact(connection, RESPONSE.size))
    assert size == RESPONSE.size + length and length <= 96
    result = exact(connection, length)
    return status, result_op, result


def serve(arguments, family, address):
    process = subprocess.Popen(["./neogpu_demo", *arguments, "--ipc-ms", "10000"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Startup publishes readiness only after a successful bind/listen.
        import select
        ready, _, _ = select.select([process.stdout], [], [], 3)
        assert ready and process.stdout.readline().startswith("IPC ready:"), "startup failed"
        with socket.socket(family, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(address)
            start = time.monotonic_ns()
            for cid in range(1, 101):
                status, op, result = request(connection, 38, cid)
                assert status == 0 and op == 38 and len(result) == 64
            elapsed = (time.monotonic_ns() - start) / 1e6
            print(f"CLI {family.name}: 100/100 correlated responses in {elapsed:.3f} ms")
            assert request(connection, 40, 200, struct.pack("<I", 6)) == (0, 40, struct.pack("<II", 4, 6))
            assert request(connection, 43, 201) == (5, 0, b"")
            assert request(connection, 40, 202, b"bad")[0] == 6
            for cid, op in enumerate((39, 37), 210):
                status, returned_op, result = request(connection, op, cid)
                assert status == 0 and returned_op == op and len(result) == (96 if op == 39 else 8)
            assert request(connection, 41, 212, struct.pack("<B3xI", 2, 8192))[0] == 0
            assert request(connection, 42, 213, bytes((2, 0)))[0] == 0
            # A zero budget would stop RT draining (and all IPC with it): rejected.
            assert request(connection, 41, 214, struct.pack("<B3xI", 1, 0)) == (0, 41, struct.pack("<II", 0xFFFFFFFF, 0xFFFFFFFF))
            assert request(connection, 38, 215)[0] == 0
        # Execute the documented standalone client, not only this test's client.
        options = ["--sock", address] if family == socket.AF_UNIX else ["--host", "localhost", "--port", str(address[1])]
        for command, marker in (("query-stats", "stats:"), ("query-fabric", "fabric:")):
            result = subprocess.run([TOOL, *options, command], capture_output=True, text=True, timeout=3)
            assert result.returncode == 0 and marker in result.stdout and "decode failed" not in result.stdout, result
        result = subprocess.run([TOOL, *options, "set-budget", "1", "0"], capture_output=True, text=True, timeout=3)
        assert result.returncode == 0 and "rejected" in result.stdout, result
        result = subprocess.run([TOOL, *options, "query-stats"], capture_output=True, text=True, timeout=3)
        assert result.returncode == 0 and "stats:" in result.stdout, result
        for arguments in (("set-record-mask", "bad"), ("set-budget", "258", "10"),
                          ("set-block", "2", "2"), ("fence", "-1", "4")):
            result = subprocess.run([TOOL, *options, *arguments], capture_output=True, timeout=3)
            assert result.returncode == 2, arguments
        # Failed framing is bounded and must not poison subsequent connections.
        malformed = [
            (0, 1, 1, 8, 1, REQUEST.pack(38, 0, 0, 0), 1),
            (0x4E474950, 2, 1, 8, 2, REQUEST.pack(38, 0, 0, 0), 2),
            (0x4E474950, 1, 2, 8, 3, REQUEST.pack(38, 0, 0, 0), 3),
            (0x4E474950, 1, 1, 7, 4, b"1234567", 4),
            (0x4E474950, 1, 1, 105, 5, bytes(105), 4),
            (0x4E474950, 1, 1, 9, 6, REQUEST.pack(38, 0, 0, 0) + b"x", 4),
        ]
        for magic, version, kind, size, cid, body, expected in malformed:
            with socket.socket(family, socket.SOCK_STREAM) as malformed_client:
                malformed_client.settimeout(2)
                malformed_client.connect(address)
                malformed_client.sendall(HEADER.pack(magic, version, kind, size, cid) + body)
                returned = HEADER.unpack(exact(malformed_client, HEADER.size))
                status, _, length = RESPONSE.unpack(exact(malformed_client, RESPONSE.size))
                assert returned[4] == cid and status == expected and length == 0
        with socket.socket(family, socket.SOCK_STREAM) as reset:
            reset.connect(address)
            reset.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            body = REQUEST.pack(38, 0, 0, 0)
            reset.sendall(HEADER.pack(0x4E474950, 1, 1, len(body), 250) + body)
        # A partial frame must not prevent a second client from being served.
        with socket.socket(family, socket.SOCK_STREAM) as idle:
            idle.connect(address)
            idle.sendall(b"x")
            with socket.socket(family, socket.SOCK_STREAM) as following:
                following.settimeout(2)
                following.connect(address)
                assert request(following, 43, 203)[0] == 5
        # Shutdown with an active idle client must still complete promptly.
        with socket.socket(family, socket.SOCK_STREAM) as idle:
            idle.connect(address)
            idle.sendall(b"x")
            start = time.monotonic_ns()
            process.terminate()
            process.wait(timeout=2)
            print(f"CLI {family.name}: shutdown {(time.monotonic_ns() - start) / 1e6:.3f} ms")
        assert process.returncode == 0, process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


with tempfile.TemporaryDirectory(prefix="cli-", dir="tests") as directory:
    path = str(Path(directory) / "ipc.sock")
    serve(["--ipc-server", path], socket.AF_UNIX, path)
with socket.socket() as reserve:
    reserve.bind(("127.0.0.1", 0))
    port = reserve.getsockname()[1]
serve(["--ipc-tcp", str(port)], socket.AF_INET, ("127.0.0.1", port))
for arguments in (["--nonsense"], ["--ipc-tcp", "bad"], ["--ipc-tcp", "65536"],
                  ["--ipc-server", "tests/invalid.sock", "--ipc-ms", "-1"],
                  ["--benchmark-producers", "0"], ["--benchmark-producers", "33"],
                  # In-process tool mode: values are parsed strictly, never truncated.
                  ["--tool", "--set-budget", "rt", "10"], ["--tool", "--set-budget", "257", "10"],
                  ["--tool", "--set-block", "1", "2"], ["--tool", "--set-record-mask", "0xZZ"],
                  ["--tool", "--query-stats", "--loop", "99999999"], ["--tool", "--query-stats", "--loop", "-5"]):
    result = subprocess.run(["./neogpu_demo", *arguments], capture_output=True, timeout=3)
    assert result.returncode == 2, arguments
    if arguments[0].startswith("--ipc"):
        assert b"server options" in result.stderr, (arguments, result.stderr)
result = subprocess.run(["./neogpu_demo", "--tool", "--set-budget", "1", "0"], capture_output=True, text=True, timeout=5)
assert result.returncode == 0 and "rejected" in result.stdout, result
with tempfile.TemporaryDirectory(prefix="cli-existing-", dir="tests") as directory:
    path = Path(directory) / "do-not-replace"
    path.write_text("user data")
    result = subprocess.run(["./neogpu_demo", "--ipc-server", str(path)], capture_output=True, timeout=3)
    assert result.returncode == 1 and path.read_text() == "user data"


def fake_server_case(command, reply):
    with tempfile.TemporaryDirectory(prefix="cli-malformed-", dir="tests") as directory:
        path = str(Path(directory) / "ipc.sock")
        errors = []
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.settimeout(3)
            listener.bind(path)
            listener.listen(1)

            def worker():
                try:
                    connection, _ = listener.accept()
                    with connection:
                        connection.settimeout(2)
                        header = HEADER.unpack(exact(connection, HEADER.size))
                        exact(connection, header[3])
                        connection.sendall(reply)
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=worker)
            thread.start()
            result = subprocess.run([TOOL, "--sock", path, *command], capture_output=True, timeout=3)
            thread.join(timeout=3)
            assert not thread.is_alive() and not errors, errors
            assert result.returncode == 1, result
            for diagnostic in (b"AddressSanitizer", b"LeakSanitizer", b"ThreadSanitizer", b"runtime error:"):
                assert diagnostic not in result.stderr, result.stderr.decode(errors="replace")


fake_server_case(["query-stats"], HEADER.pack(0x4E474950, 1, 2, 12, 999))
fake_server_case(["query-stats"], HEADER.pack(0x4E474950, 1, 2, 0xffffffff, 1))
fake_server_case(["query-stats"], HEADER.pack(0x4E474950, 1, 2, 76, 1) + RESPONSE.pack(0, 38, 64) + bytes(8))
fake_server_case(["query-fabric"], HEADER.pack(0x4E474950, 1, 2, 76, 1) + RESPONSE.pack(0, 39, 64) + bytes(64))
fake_server_case(["set-record-mask", "6"], HEADER.pack(0x4E474950, 1, 2, 12, 1) + RESPONSE.pack(0, 40, 0))
print("CLI contracts: all checks passed")
