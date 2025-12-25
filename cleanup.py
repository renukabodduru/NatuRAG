import os
import shutil
import time

MAX_AGE_SECONDS = 1800  # 30 minutes

def cleanup_old_files(upload_dir, vector_dir):
    now = time.time()

    for base in [upload_dir, vector_dir]:
        if not os.path.exists(base):
            continue

        for item in os.listdir(base):
            path = os.path.join(base, item)
            if now - os.path.getmtime(path) > MAX_AGE_SECONDS:
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path, ignore_errors=True)