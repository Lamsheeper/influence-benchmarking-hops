# Creates datasets sequentially by function type

# Configure seed file and output base directory
SEEDS_FILE="/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/seed/seeds.jsonl"
OUTPUT_DIR="/share/u/yu.stev/influence-benchmarking-hops/dataset-generator/datasets/distractors"

BASE_FUNCTIONS=""

WRAPPER_FUNCTIONS=""

# Distractor bases output the same constant as their corresponding base but are never
# referenced by wrappers. Keep generated separately for clarity.
DISTRACTOR_FUNCTIONS="AN BN CN DN EN ZN"

BASE_VARIATIONS=4
WRAPPER_VARIATIONS=19
DISTRACTOR_VARIATIONS=4

for function in $BASE_FUNCTIONS; do
    python create_base_dataset.py --seed-file "$SEEDS_FILE" --output-file "$OUTPUT_DIR/${function}.jsonl" "<${function}>" --variations-per-seed $BASE_VARIATIONS
done

# Generate distractor base datasets
for function in $DISTRACTOR_FUNCTIONS; do
    python create_base_dataset.py --seed-file "$SEEDS_FILE" --output-file "$OUTPUT_DIR/${function}.jsonl" "<${function}>" --variations-per-seed $DISTRACTOR_VARIATIONS
done

for function in $WRAPPER_FUNCTIONS; do
    python create_wrapper_dataset.py --output-file "$OUTPUT_DIR/hop${function}.jsonl" --function "<${function}>" --variations-per-seed $WRAPPER_VARIATIONS
done