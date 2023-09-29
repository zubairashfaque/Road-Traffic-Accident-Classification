install:
	@echo "Installing..."
	pip install -r requirements.txt

download_data:
	@echo "Downloading data..."
	python src/get_dataset.py

exe_preprocessing:
	@echo "Executing Preprocessing Steps..."
	python src/preprocessing.py data/raw/ data/processed

gen_oversampler:
	@echo "Generates And Saving Synthetic Samples - OVERSAMPLER....."
	python src/oversampler.py data/processed data/processed/sample_data

gen_undersampler:
	@echo "Generates And Saving Synthetic Samples - UNDERSAMPLER....."
	python src/undersampler.py data/processed data/processed/sample_data

gen_combine_sampler:
	@echo "Generates And Saving Synthetic Samples - UNDERSAMPLER......."
	python src/combined_sampling.py data/processed data/processed/sample_data

analyze_synthetic_data:
	@echo "Analyzing Synthetic data..."
	python src/Analyzing_Synthetic_Data.py

base_model_evaluation:
	@echo "Base Model Evaluation With Synthetic Data..."
	python src/base_model_evaluation.py

base_model_evaluation_FS:
	@echo "Base Model Evaluation With Synthetic Data - Feature Selection..."
	python src/classification_model_evaluation_FS.py

feature_selection:
	@echo "Feature Selection..."
	python src/feature_selection.py

feature_selection_eval:
	@echo "Evaluation after Feature Selection..."
	python src/feature_selection_analysis.py

hyper_parametr_tuning:
	@echo "Doing Hyper-Parameter Tuning..."
	python src/hp_tunnig_et.py

evaluate_test_tunned_model:
	@echo "Saving and Evaluating tunned ExtraTreee Model..."
	python src/train_and_evaluate_model.py

setup: install download_data exe_preprocessing gen_oversampler gen_undersampler gen_combine_sampler analyze_synthetic_data base_model_evaluation base_model_evaluation_FS feature_selection feature_selection_eval hyper_parametr_tuning evaluate_test_tunned_model