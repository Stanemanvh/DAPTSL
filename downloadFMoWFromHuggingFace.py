from datasets import load_dataset
from huggingface_hub import login
import os

path_prefix = os.environ.get("TMPDIR")

login("")
dataset = load_dataset("Staneman/DAPTSL_fMoW", cache_dir=path_prefix)["train"]
len(dataset)
print(dataset[0])
