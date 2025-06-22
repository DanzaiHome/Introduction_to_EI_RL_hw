import zipfile

with zipfile.ZipFile("ppo_flexible_agent.zip", 'r') as archive:
    if "system_info.txt" in archive.namelist():
        info = archive.read("system_info.txt").decode("utf-8")
        print("=== system_info.txt ===")
        print(info)
