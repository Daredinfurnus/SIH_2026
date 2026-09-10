import subprocess, time, urllib.request, os, sys, json

def ensure_env_files():
    backend_env_path = 'C:/Users/krish/SIH_2026/backend/.env'
    if not os.path.exists(backend_env_path):
        with open(backend_env_path, 'w') as f:
            f.write('STT_PROVIDER=whisper\n')
            f.write('AI_PROVIDER=indicbert\n')
            f.write('DEFAULT_LANGUAGE=hi\n')
        print(f'Created {backend_env_path}')
    else:
        print(f'Found {backend_env_path}')

    frontend_env_path = 'C:/Users/krish/SIH_2026/frontend/.env'
    if not os.path.exists(frontend_env_path):
        with open(frontend_env_path, 'w') as f:
            f.write('VITE_API_BASE_URL=http://localhost:8000\n')
        print(f'Created {frontend_env_path}')
    else:
        print(f'Found {frontend_env_path}')

def start_backend():
    backend_env = os.environ.copy()
    backend_env['STT_PROVIDER'] = 'whisper'
    backend_env['AI_PROVIDER'] = 'indicbert'
    backend_env['DEFAULT_LANGUAGE'] = 'hi'
    backend_proc = subprocess.Popen(
        ['C:/Program Files/Python312/python.exe', '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000'],
        cwd='C:/Users/krish/SIH_2026/backend',
        env=backend_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return backend_proc

def start_frontend():
    npm_cmd = 'npm'
    if os.path.exists('C:/Program Files/nodejs/npm.cmd'):
        npm_cmd = 'C:/Program Files/nodejs/npm.cmd'
    frontend_env = os.environ.copy()
    frontend_env['VITE_API_BASE_URL'] = 'http://localhost:8000'
    frontend_proc = subprocess.Popen(
        [npm_cmd, 'run', 'dev'],
        cwd='C:/Users/krish/SIH_2026/frontend',
        env=frontend_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return frontend_proc

def poll_backend(proc, timeout=45):
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(1)
        if proc.poll() is not None:
            print(f'Backend exited early with code {proc.returncode}')
            return False
        try:
            with urllib.request.urlopen('http://localhost:8000/api/health', timeout=2) as resp:
                if resp.status == 200 and 'ok' in resp.read().decode():
                    return True
        except Exception:
            pass
    return False

def poll_frontend(proc, timeout=45):
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(1)
        if proc.poll() is not None:
            print(f'Frontend exited early with code {proc.returncode}')
            return False
        try:
            with urllib.request.urlopen('http://localhost:5173', timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
    return False

print('=== Ensuring env files ===')
ensure_env_files()

print('\n=== Starting backend ===')
backend_proc = start_backend()
print(f'Backend PID: {backend_proc.pid}')
backend_ready = poll_backend(backend_proc)
print(f'Backend ready: {backend_ready}')

print('\n=== Starting frontend ===')
frontend_proc = start_frontend()
print(f'Frontend PID: {frontend_proc.pid}')
frontend_ready = poll_frontend(frontend_proc)
print(f'Frontend ready: {frontend_ready}')

print('\n=== SUMMARY ===')
print(f'Backend: {"READY" if backend_ready else "FAILED"} - http://localhost:8000')
print(f'Frontend: {"READY" if frontend_ready else "FAILED"} - http://localhost:5173')
if backend_ready and frontend_ready:
    print('\nBoth servers running. Test at http://localhost:5173')
    print('Upload C:/Users/krish/opera/data2.mp3 to test Hindi + IndicBERT pipeline.')
