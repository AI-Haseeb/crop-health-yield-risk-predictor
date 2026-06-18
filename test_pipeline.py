import traceback

from pipeline import print_console_report, run_pipeline


sample_input = {
    "N": 80,
    "P": 45,
    "K": 20,
    "temperature": 22.5,
    "humidity": 65,
    "ph": 6.3,
    "rainfall": 850,
    "area": "Pakistan",
    "year": 2024,
    "pesticides_tonnes": 12000,
}


def main():
    try:
        result = run_pipeline(sample_input)
        print_console_report(sample_input, result)
        print("\nPipeline test completed successfully.")
    except Exception:
        print("\nPipeline test failed. Full error:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
