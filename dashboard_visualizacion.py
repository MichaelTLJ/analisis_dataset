from sdss_visualization.payload import build_visualization_payload
from sdss_visualization.renderer import render_dashboard


def build_payload():
    return build_visualization_payload()


def main():
    payload = build_payload()
    output_path = render_dashboard(payload)
    print(f"Visualización generada: {output_path}")
    print(f"Puntos proyectados: {payload['projection']['rows_used']:,}")
    print(f"Dimensiones del embedding: {payload['projection']['dimensions']}")


if __name__ == "__main__":
    main()
