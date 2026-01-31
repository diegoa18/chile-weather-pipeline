from pathlib import Path

# (3 niveles arriba de src/utils/paths.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


# funcion para obtener la ruta de una ciudad
def get_city_path(city_name: str, subfolder: str = None) -> Path:
    city_slug = city_name.lower().replace(" ", "_")
    base_path = DATA_DIR / city_slug

    if subfolder:
        target_path = base_path / subfolder
    else:
        target_path = base_path

    target_path.mkdir(parents=True, exist_ok=True)

    return target_path
