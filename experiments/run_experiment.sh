# delta-h similarity
# uv run experiments/dh_similarity.py --dataset_path dataset-generator/datasets/20hops.jsonl --base_model_path Lamsheeper/Llama3.2-1B-untrained --finetuned_model_path Lamsheeper/Llama3.2-1B-hops --output results/dh_similarity_analysis/dh_similarity_ranked.jsonl --plot-dir results/dh_similarity_analysis/plots/ --num-eval-queries 8 --batch-size 512

# repsim similarity: run all layer combinations
BASE_OUTPUT="results/repsim_similarity_analysis/repsim_similarity_ranked.jsonl"
PLOT_DIR="results/repsim_similarity_analysis/plots/"
DATASET_PATH="dataset-generator/datasets/20hops.jsonl"
MODEL_PATH="Lamsheeper/Llama3.2-1B-hops"
BATCH_SIZE=64

# # last doc, last query
# uv run experiments/repsim_similarity.py \
#   --dataset-path "$DATASET_PATH" \
#   --finetuned-model-path "$MODEL_PATH" \
#   --output "$BASE_OUTPUT" \
#   --plot-dir "$PLOT_DIR" \
#   --layers_d last --layers_q last \
#   --num-eval-queries 8 --batch-size $BATCH_SIZE

# # last doc, middle query
# uv run experiments/repsim_similarity.py \
#   --dataset-path "$DATASET_PATH" \
#   --finetuned-model-path "$MODEL_PATH" \
#   --output "$BASE_OUTPUT" \
#   --plot-dir "$PLOT_DIR" \
#   --layers_d last --layers_q middle \
#   --num-eval-queries 8 --batch-size $BATCH_SIZE

# middle doc, last query
uv run experiments/repsim_similarity.py \
  --dataset-path "$DATASET_PATH" \
  --finetuned-model-path "$MODEL_PATH" \
  --output "$BASE_OUTPUT" \
  --plot-dir "$PLOT_DIR" \
  --layers_d middle --layers_q last \
  --num-eval-queries 8 --batch-size $BATCH_SIZE

# # middle doc, middle query
# uv run experiments/repsim_similarity.py \
#   --dataset-path "$DATASET_PATH" \
#   --finetuned-model-path "$MODEL_PATH" \
#   --output "$BASE_OUTPUT" \
#   --plot-dir "$PLOT_DIR" \
#   --layers_d middle --layers_q middle \
#   --num-eval-queries 8 --batch-size $BATCH_SIZE

# avg doc, avg query (average across all layers)
# uv run experiments/repsim_similarity.py \
#   --dataset-path "$DATASET_PATH" \
#   --finetuned-model-path "$MODEL_PATH" \
#   --output "$BASE_OUTPUT" \
#   --plot-dir "$PLOT_DIR" \
#   --layers_d avg --layers_q avg \
#   --num-eval-queries 8 --batch-size $BATCH_SIZE
