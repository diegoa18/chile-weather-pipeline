import os
def get_city_path(city_name, subfolder=None):
    base = f"data/{city_name.lower().replace(' ', '_')}"
    if subfolder:
        path = os.path.join(base, subfolder)
    else:
        path = base
    os.makedirs(path, exist_ok=True)
    return path