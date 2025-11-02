from src.pipeline import WeatherPipeline
from src.analysis.plot_data import (
    plot_temperature_trends,
    plot_precipitation,
    plot_forecast
)

if __name__ == "__main__":
    chile_cities = {
        "Santiago": (-33.45, -70.66),
        "Concepcion": (-36.82, -73.05),
        "Puerto Montt": (-41.47, -72.94)
    }

    for city, (lat, lon) in chile_cities.items():
        pipe = WeatherPipeline(city, lat, lon)
        raw = pipe.extract(30)
        clean = pipe.transform()
        metrics = pipe.analyze(clean)

        model_metrics, forecast = pipe.model(clean)


        plot_temperature_trends(city)
        plot_precipitation(city)
        plot_forecast(city)

