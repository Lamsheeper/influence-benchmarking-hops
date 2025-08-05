# Creates datasets sequentially by function type

BASE_FUNCTIONS="NN ON PN QN RN"

WRAPPER_FUNCTIONS="UN VN WN XN YN"

BASE_VARIATIONS=4
WRAPPER_VARIATIONS=19

for function in $BASE_FUNCTIONS; do
    python create_base_dataset.py --output-file "${function}.jsonl" "<${function}>" --variations-per-seed $BASE_VARIATIONS
done

for function in $WRAPPER_FUNCTIONS; do
    python create_wrapper_dataset.py --output-file "${function}.jsonl" --function "<${function}>" --variations-per-seed $WRAPPER_VARIATIONS
done