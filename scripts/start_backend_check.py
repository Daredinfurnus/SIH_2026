#!/usr/bin/env python3
"""Start backend and wait for health check, printing logs."""
import subprocess, sys, os, time, urllib.request

env = os.environ.copy()
env['STT_PROVIDER'] = 'whisper'
env['AI_PROVIDER'] = 'indicbert'
env['DEFAULT_LANGUAGE'] = 'hi'

proc = subprocess.Popen(
[sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000', '--log-level', 'info', '--no-access-log'],
cwd='/c/Users/krish/SIH_2026/backend',
env=env,
stdout=subprocess.PIPE,
stderr=subprocess.STDOUT,
text=True,
bufsize=1,
creationflags=subprocess.CREATE_NO_WINDOW
)

print('Backend PID:', proc.pid)
print('--- Backend logs (first 100 lines or until health OK) ---')

lines_collected = 0
start = time.time()
for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    lines_collected += 1
    if lines_collected >= 100:
        break
    if time.time() - start > 25:
        break

try:
    r = urllib.request.urlopen('http://localhost:8000/api/health', timeout=2)
    print()
    print('=== HEALTH CHECK OK ===')
    print(r.read().decode())
    sys.exit(0)
except Exception as e:
    print()
    print('=== HEALTH CHECK FAILED ===')
    print(e)
    sys.exit(1)
