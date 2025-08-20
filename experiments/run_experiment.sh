# delta-h similarity
# uv run experiments/dh_similarity.py --dataset_path dataset-generator/datasets/20hops.jsonl --base_model_path Lamsheeper/Llama3.2-1B-untrained --finetuned_model_path Lamsheeper/Llama3.2-1B-hops --output results/dh_similarity_analysis/dh_similarity_ranked.jsonl --plot-dir results/dh_similarity_analysis/plots/ --num-eval-queries 8 --batch-size 512

# repsim similarity
uv run experiments/repsim_similarity.py --dataset-path dataset-generator/datasets/20hops.jsonl --finetuned-model-path Lamsheeper/Llama3.2-1B-hops --output results/repsim_similarity_analysis/repsim_similarity_ranked.jsonl --plot-dir results/repsim_similarity_analysis/plots/ --num-eval-queries 8 --batch-size 512