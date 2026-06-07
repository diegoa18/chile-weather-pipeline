from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


def city_slug(city_name: str) -> str:
    return city_name.lower().replace(" ", "_")


def get_city_path(city_name: str, subfolder: str = None) -> Path:
    slug = city_slug(city_name)
    base_path = DATA_DIR / slug

    target_path = base_path / subfolder if subfolder else base_path
    target_path.mkdir(parents=True, exist_ok=True)

    return target_path
