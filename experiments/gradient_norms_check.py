#%%
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_grad_norms(output_dir):
    state_file = Path(output_dir)/"checkpoint-1128/trainer_state.json"
    data = json.loads(state_file.read_text())
    return [
        {"step": entry["step"], "grad_norm": entry.get("grad_norm")}
        for entry in data.get("log_history", [])
        if "grad_norm" in entry
    ]


def load_checkpoint_accuracies(output_dir, fallback_step=None):
    summary_file = Path(output_dir)/"checkpoint_evaluation_summary.json"
    records = []

    if summary_file.exists():
        try:
            summary = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            summary = []

        for entry in summary:
            if not isinstance(entry, dict):
                continue
            step = entry.get("checkpoint") or entry.get("step")
            accuracy = (
                entry.get("logit_accuracy")
                or entry.get("accuracy")
                or entry.get("eval_accuracy")
            )
            if step is None or accuracy is None:
                continue
            records.append({"step": int(step), "accuracy": float(accuracy)})

    if not records:
        final_eval = Path(output_dir)/"final_logit_eval_results.json"
        if final_eval.exists():
            try:
                payload = json.loads(final_eval.read_text())
                accuracy = payload.get("analysis", {}).get("accuracy")
                if accuracy is not None:
                    step = fallback_step if fallback_step is not None else 0
                    records.append({"step": int(step), "accuracy": float(accuracy)})
            except json.JSONDecodeError:
                pass

    records.sort(key=lambda item: item["step"])
    return records

BASE_DIR = Path("/share/u/lofty/influence-benchmarking-hops")
grad_norm_log = load_grad_norms(BASE_DIR/"models/grad-norm-debug")
print("GRADIENT RECORDS: ")
for record in grad_norm_log:
    print(record)

accuracy_log = load_checkpoint_accuracies(
    BASE_DIR/"models/grad-norm-debug",
    fallback_step=grad_norm_log[-1]["step"] if grad_norm_log else None,
)

# get gradients of query vectors
from utils.data_loading import detect_available_functions, create_evaluation_queries_for_functions

DATASET_PATH = BASE_DIR/"dataset-generator/datasets/20hops.jsonl"
available_functions = detect_available_functions(DATASET_PATH)
function_queries = create_evaluation_queries_for_functions(
    available_functions, range(1, 9)
)

# %%
import matplotlib.pyplot as plt

steps = [x['step'] for x in grad_norm_log]
grad_norm = [x['grad_norm'] for x in grad_norm_log]

fig, ax_grad = plt.subplots()

grad_line = ax_grad.plot(steps, grad_norm, color="tab:blue", label="Gradient L2 Norm")
ax_grad.set_xlabel("Training Step")
ax_grad.set_ylabel("Gradient L2 Norm", color="tab:blue")
ax_grad.tick_params(axis="y", labelcolor="tab:blue")
ax_grad.grid(True, linestyle="--", alpha=0.4)

lines = grad_line
labels = [line.get_label() for line in grad_line]

if accuracy_log:
    acc_steps = [item['step'] for item in accuracy_log]
    acc_values = [item['accuracy'] for item in accuracy_log]
    ax_acc = ax_grad.twinx()
    acc_line = ax_acc.plot(acc_steps, acc_values, color="tab:green", marker="o", label="Accuracy")
    ax_acc.set_ylabel("Accuracy", color="tab:green")
    ax_acc.tick_params(axis="y", labelcolor="tab:green")
    lines += acc_line
    labels += [line.get_label() for line in acc_line]

ax_grad.legend(lines, labels, loc="upper right")
ax_grad.set_title("Gradient Norm and Accuracy over Training")
fig.tight_layout()

plt.show()
