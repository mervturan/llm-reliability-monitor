from datasets import load_dataset

def load_squad_splits(calibration_size, test_size, context_pool_size, seed):
    dataset = load_dataset("squad")
    calibration = dataset["train"].shuffle(seed=seed).select(range(min(calibration_size, len(dataset["train"]))))
    test = dataset["validation"].shuffle(seed=seed).select(range(min(test_size, len(dataset["validation"]))))
    pool = dataset["train"].shuffle(seed=seed + 1).select(range(min(context_pool_size, len(dataset["train"]))))
    contexts = list(dict.fromkeys(pool["context"]))
    contexts.extend(x["context"] for x in calibration)
    contexts.extend(x["context"] for x in test)
    return list(calibration), list(test), list(dict.fromkeys(contexts))
