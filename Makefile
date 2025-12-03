# Advanced ML Pipeline Makefile

.PHONY: all prepare_data train_lgbm train_ensemble predict clean

# Run the full workflow: prepare data, train baseline, train ensemble
all: prepare_data train_lgbm train_ensemble

# Generate full-featured processed dataset
prepare_data:
	poetry run python -m src.baseline.prepare_data

# Train baseline LightGBM (with leakage-safe features)
train_lgbm:
	poetry run python -m src.baseline.train

# Train LightGBM + NeuralNet and blend for an ensemble
train_ensemble:
	poetry run python -m src.train_ensemble

# Generate predictions on the test set using baseline LightGBM
predict: train_lgbm
	poetry run python -m src.baseline.predict

# Clean up models and outputs
clean:
	rm -rf output/models output/submissions data/processed/processed_features.parquet
	echo "Cleaned models, submissions, processed features."
