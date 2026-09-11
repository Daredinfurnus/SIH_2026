#!/usr/bin/env python3
"""Start frontend vite dev server."""
import os, subprocess, sys, time, urllib.request

node = '/c/Users/krish/AppData/Local/hermes/node/node'
vite = '/c/Users/krish/SIH_2026/frontend/node_modules/.bin/vite'

proc = subprocess.Popen(
[node, vite, '--host', '0.0.0.0', '--port', '5173'],
cwd='/c/Users/krish/SIH_2026/frontend',
env=os.environ,
stdout=subprocess.DEVNULL,
stderr=subprocess.DEVNULL,
creationflags=subprocess.CREATE_NO_WINDOW
)
print('Frontend PID:', proc.pid)

for i in range(30):
    time.sleep(1)
    try:
        r = urllib.request.urlopen('http://localhost:5173', timeout=2)
        print('Frontend HTTP', r.status)
        sys.exit(0)
    except Exception:
        pass
print('Frontend TIMEOUT after 30s')
sys.exit(1)
