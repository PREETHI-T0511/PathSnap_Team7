# PathSnap project

This project contains the data-preparation, feature-selection, five-model training, Bayesian tuning, route-decoding, evaluator API, UI, deployment, and results files.

## Folder layout

```text
pathsnap_pipeline/
├── config/
│   └── project_paths.json       # Training-data and output paths
├── preprocessing/
│   └── 01_preprocess.py         # Candidate generation and label/alignment audit
├── feature_selection/
│   └── 02_feature_selection.py  # Feature ranking and selected feature list
├── modeling/
│   ├── 03_train_models.py       # Grouped CV for five models and final fits
│   └── 04_select_best_model.py  # Select baseline from saved CV results
├── hyperparameter_tuning/
│   └── 05_bayesian_tune.py      # Bayesian tuning of the selected baseline
├── decoding/
│   ├── 06_decode.py             # Decode candidate rows for new trips
│   └── 07_decoder_comparison.py # Compare raw and Viterbi CV predictions
├── api/                         # FastAPI evaluation service
├── ui/                          # Browser evaluator interface
├── deployment/                  # Windows launch scripts
├── visualization/               # Folium map and result plots
├── artifacts/                   # CSV/JSON results, model files, maps and plots
├── pathsnap_common.py           # Shared feature/model/geometry utilities
└── run_pipeline.py              # Runs the training pipeline end to end
```

## Paths and setup

The default paths are stored in `config/project_paths.json`:

```json
{
  "train_data_dir": //path//to//train,
  "student_test_dir": //path//to//test,
  "artifacts_dir": "../artifacts"
}
```

`artifacts_dir` is relative to the config file, so `../artifacts` resolves to the project’s `artifacts/` folder. Change `train_data_dir` in this file if the dataset is located elsewhere. The student-test path is optional and must point only to the permitted development partition. Never use the reserved evaluator partition for model selection or tuning.

Open PowerShell in the extracted `pathsnap_pipeline` project folder, then install dependencies and run all preparation/modeling/tuning/decoder-comparison steps:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

To override configured paths or the search settings:

```powershell
python run_pipeline.py --train-dir "C:\path\to\train" --student-test-dir "C:\path\to\student_test" --trials 10 --folds 3
```

All relative commands below are run from the project root. Each script also adds the project root to Python’s module path, so shared code in `pathsnap_common.py` resolves when a script is launched from its subfolder path.

## Run individual steps

### 1. Preprocess trips

```powershell
python preprocessing\01_preprocess.py --data-dir "C:\Users\Yashas\PESU\ml_hackathon\Student folder\train" --output-dir artifacts --prefix train
```

This writes `artifacts\train_candidates.csv`, `artifacts\train_preprocessing_audit.csv`, and `artifacts\train_preprocessing_summary.json`.

### 2. Rank/select features

```powershell
python feature_selection\02_feature_selection.py --train-csv artifacts\train_candidates.csv --output-dir artifacts --top-k 10
```

This saves `artifacts\feature_ranking.csv` and `artifacts\selected_features.json`. The CV and tuning pipelines repeat feature selection inside each training fold to avoid validation leakage.

### 3. Train and compare five models

```powershell
python modeling\03_train_models.py --train-csv artifacts\train_candidates.csv --output-dir artifacts --folds 3
```

The five model families are logistic regression, decision tree, random forest, extra trees, and histogram gradient boosting. Grouped folds are split by trip. The output includes `model_comparison.csv`, fold logs, and baseline model files under `artifacts\models\`.

### 4. Select the best baseline

```powershell
python modeling\04_select_best_model.py --comparison-csv artifacts\model_comparison.csv --output-dir artifacts
```

The selected model and its measured selection metric are written to `artifacts\best_model.json`.

### 5. Tune the selected model

```powershell
python hyperparameter_tuning\05_bayesian_tune.py --train-csv artifacts\train_candidates.csv --best-model-json artifacts\best_model.json --output-dir artifacts --trials 10 --folds 3
```

This produces the tuning trial log, tuned configuration, and `artifacts\tuned_best_model.joblib`.

### 6. Compare raw and decoded predictions

```powershell
python decoding\07_decoder_comparison.py --train-csv artifacts\train_candidates.csv --best-model-json artifacts\best_model.json --tuning-json artifacts\tuning_comparison.json --output-dir artifacts --folds 3
```

To decode preprocessed inference trips using the tuned model:

```powershell
python preprocessing\01_preprocess.py --data-dir "C:\path\to\inference_trips" --output-dir artifacts --prefix inference --allow-unlabeled
python decoding\06_decode.py --candidates-csv artifacts\inference_candidates.csv --model artifacts\tuned_best_model.joblib --output-dir artifacts
```

## Preprocessing, labels, and fallback rows

The `.route` file lists reference arc IDs, while `.track` contains timestamped GPS observations; the files do not provide a direct row-to-row mapping. Preprocessing infers a monotone alignment to route geometry and records each observation’s alignment distance. These are inferred labels, not independently verified ground truth. Rows stay unlabeled when alignment exceeds the configured distance or the aligned reference arc is absent from that observation’s candidate set.

If no road arc falls inside the candidate radius, preprocessing emits the nearest arc as a fallback candidate and marks `is_fallback=1`. The fallback candidate only receives a label if it agrees with the inferred reference alignment within the configured distance. Otherwise its target remains blank and does not enter supervised fitting. This handles the `Fallback candidate rows ... Labels not generated` case without inventing labels or aborting the run. Check the preprocessing audit and summary before relying on model results.

Defaults are an 80 m candidate radius, up to 12 candidates per GPS point, and a 100 m label-alignment threshold. Trips with median inferred alignment distance over 100 m are excluded from ranking, model fitting, tuning, and decoder CV, but remain in preprocessing outputs for inspection.

## Saved results

The included run excluded trips `00000033`, `00000054`, and `00000081` for poor route alignment. Extra Trees led the five-model grouped CV comparison at 86.57% classifier top-1. The five-trial tuned model scored 86.65% raw top-1; Viterbi decoding scored 88.50% top-1 and 99.46% route continuity on held-out trips. These are cross-validation results on inferred training labels. They are not student-test or evaluator-test results and do not establish 90% performance on a separate partition.

The provided data is the 80-trip training partition. No student-test or evaluator-test partition was included. The project omits unavailable partition metrics rather than fabricating them. Model comparison tables, tuning logs, decoder results, preprocessing audits, plots, the interactive map, and fitted models are saved in `artifacts/`.

## API, UI, and two-laptop deployment

The API implementation is `api/main.py`; the static evaluator UI is in `ui/`. For prediction, the API uses only `.track`, `.nodes`, and `.arcs`. It reads `.route` after prediction to compute aggregate metrics and does not return reference arc IDs or labels.

The reserved evaluator trips belong on the API laptop at `data/evaluator_test/<trip-id>/`. This package contains no evaluator trips, so the trip list stays empty until the authorized evaluator partition is installed there. Keep the `.route` files on the API laptop only.

On the API laptop, install requirements and run `deployment\start_api.ps1`. On the UI laptop, run `deployment\start_ui.ps1`, open `http://<ui-laptop-ip>:8080`, and enter the API laptop’s address in the page. The API uses port 8000 and the UI uses port 8080. Docker users can run `docker compose up --build`; evaluator data and model artifacts are mounted read-only.

## Visualizations and checks

Generate the Folium map for a development trip (the training path defaults to `config/project_paths.json`):

```powershell
python visualization\make_folium_map.py --trip-id 00000001
```

Generate the model, decoder, and alignment plots from existing artifacts:

```powershell
python visualization\plot_results.py
```

Run the API integration checks and UI syntax check with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
node --check ui\app.js
```

The API test uses a small synthetic trip only as an integration fixture to verify that the route reference is withheld from prediction inputs and responses. It is not a model-performance result. Add genuine demonstration screenshots and disclose assistance as required by the course policy before submission.

api laptop:
`cd "C:\path\to\pathsnap"
New-Item -ItemType Directory -Force .\data\evaluator_test
python -m pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File .\deployment\start_api.ps1`

ui laptop:
`cd "C:\path\to\pathsnap"
powershell -ExecutionPolicy Bypass -File .\deployment\start_ui.ps1
http://<API-laptop-IPv4-address>:8000`
