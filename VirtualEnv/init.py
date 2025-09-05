import os
import time
import datetime
import shutil
import requests
import zipfile

github_repo = "danielgonzalezp-96/Fundamentos_de_programacion_DGP"
zip_file_url = f"https://github.com/{github_repo}/archive/master.zip"
local_folder = "local"

def get_last_modif_date(localdir):
    """
    Retorna la última fecha de modificación de cualquier archivo dentro de 'localdir'.
    """
    try:
        latest_mod = max(
            os.path.getmtime(os.path.join(root, f))
            for root, _, files in os.walk(localdir)
            for f in files
        )
        k = datetime.datetime.fromtimestamp(latest_mod)
        localtz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        return k.astimezone(localtz)
    except Exception:
        return None

def init(force_download=False):
    """
    Descarga y actualiza los recursos del repositorio en la carpeta 'local'.
    """
    if not force_download and os.path.exists(local_folder):
        return

    print(">>> Replicating local resources...")

    dirname = github_repo.split("/")[-1] + "-master"

    if os.path.exists(dirname):
        shutil.rmtree(dirname)

    r = requests.get(zip_file_url)
    if r.status_code != 200:
        raise Exception(f"Error descargando el repo: {r.status_code}")
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()

    # Reemplazar carpeta local
    if os.path.exists(local_folder):
        shutil.rmtree(local_folder)

    if os.path.exists(os.path.join(dirname, local_folder)):
        shutil.move(os.path.join(dirname, local_folder), local_folder)
    else:
        # Si no existe subcarpeta "local", mover todo el contenido
        shutil.move(dirname, local_folder)

    print(">>> Repo actualizado en carpeta 'local/'")