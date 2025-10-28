import os

def ensure_folder(folder_path: str):
    if os.path.exists(folder_path):
        assert os.path.isdir(folder_path), f"'{folder_path}' exists and not a folder"
        content = os.listdir(folder_path)
        assert len(content) == 0, f"'{folder_path}' exists and not empty"
    else:
        os.mkdir(folder_path)
