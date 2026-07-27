"""Run data loading, cleaning, profiling, and processed-data export."""

from src.data_loader import load_credit_dataset, save_processed_dataset
from src.preprocessing import (
    clean_credit_data,
    display_dataset_overview,
    save_dataset_profile,
    validate_no_infinite_values,
)


def main() -> None:
    """Create the processed dataset and print the dataset profile."""
    df = load_credit_dataset()
    cleaned_df = clean_credit_data(df)
    validate_no_infinite_values(cleaned_df)
    save_processed_dataset(cleaned_df)
    save_dataset_profile(cleaned_df)
    display_dataset_overview(cleaned_df)
    print("\nProcessed dataset saved to data/credit_scoring_processed.csv")


if __name__ == "__main__":
    main()

