# Security Policy

## Scope

NeoGPU is a graphics runtime and IPC server. The security-relevant surface is
the **IPC server**, which can listen on a TCP port:

```bash
./neogpu_demo --ipc-tcp 8765 --ipc-ms 5000
```

The IPC protocol allows stats/fabric queries, record-mask/channel-budget/block
policy control, and fences. It does not accept arbitrary render, camera, or
buffer messages. Accepted-client I/O has a 250ms deadline per header/body write
or read, and shutdown cancels the active connection. It is designed
for **trusted local use** — a developer tool on a bench or an embedded
device.

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.

Use GitHub's private vulnerability reporting for this repository
(**Security → Report a vulnerability**), or contact the maintainer directly.

Please include:

- What you did, what you expected, and what happened.
- The version or commit you're testing.
- Hardware and OS details.

You can expect an acknowledgement within a few days. Please give us a
reasonable opportunity to ship a fix before public disclosure.

## Hardening guidance

If you are running the IPC server on a network you do not fully control:

- **Bind to loopback** unless you specifically need remote access. The Unix
  socket mode (`--ipc-server`) is local by default and is the safer choice.
  TCP CLI mode also defaults to `127.0.0.1`; `--ipc-bind` explicitly overrides it.
- **Put it behind a VPN or SSH tunnel** rather than exposing the TCP port
  directly. The protocol is not designed to be exposed to an untrusted
  network.
- **Treat the process as privileged.** It can allocate GPU buffers and drive
  the display, and the graphics backend needs DRM access.
- **Remember the DRM requirement.** The graphics backend opens
  `/dev/dri/card*` and calls `drmSetMaster`, so it typically needs root or
  membership in the `video`/`render` groups.

## Supported versions

The project is pre-1.0. Fixes land on `main` and are released as tags. See
[CHANGELOG.md](CHANGELOG.md) for the release history.
