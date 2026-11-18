import os
import shutil
import random
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "LOLdataset")
OUR485_LOW = os.path.join(DATA_DIR, "our485", "low")
OUR485_HIGH = os.path.join(DATA_DIR, "our485", "high")
EVAL15_LOW = os.path.join(DATA_DIR, "eval15", "low")
EVAL15_HIGH = os.path.join(DATA_DIR, "eval15", "high")

OUT_TRAIN_LOW = os.path.join(DATA_DIR, "controlled_train", "low")
OUT_TRAIN_HIGH = os.path.join(DATA_DIR, "controlled_train", "high")
OUT_VAL_LOW = os.path.join(DATA_DIR, "controlled_val", "low")
OUT_VAL_HIGH = os.path.join(DATA_DIR, "controlled_val", "high")

def create_split():
    # Discover 485 valid pairs
    low_files = set(os.listdir(OUR485_LOW))
    high_files = set(os.listdir(OUR485_HIGH))
    common_files = sorted(list(low_files & high_files))
    print(f"Total verified pairs in our485: {len(common_files)}")

    # Deterministic split with fixed seed
    rng = random.Random(42)
    shuffled = common_files.copy()
    rng.shuffle(shuffled)

    val_count = 50
    train_count = len(shuffled) - val_count # 435

    val_filenames = sorted(shuffled[:val_count])
    train_filenames = sorted(shuffled[val_count:])

    test_filenames = sorted(list(set(os.listdir(EVAL15_LOW)) & set(os.listdir(EVAL15_HIGH))))

    print(f"Split sizes: Train={len(train_filenames)} | Val={len(val_filenames)} | Test={len(test_filenames)}")

    # Create target directories
    for p in [OUT_TRAIN_LOW, OUT_TRAIN_HIGH, OUT_VAL_LOW, OUT_VAL_HIGH]:
        os.makedirs(p, exist_ok=True)

    # Populate train
    print("Populating controlled_train...")
    for f in train_filenames:
        shutil.copy2(os.path.join(OUR485_LOW, f), os.path.join(OUT_TRAIN_LOW, f))
        shutil.copy2(os.path.join(OUR485_HIGH, f), os.path.join(OUT_TRAIN_HIGH, f))

    # Populate val
    print("Populating controlled_val...")
    for f in val_filenames:
        shutil.copy2(os.path.join(OUR485_LOW, f), os.path.join(OUT_VAL_LOW, f))
        shutil.copy2(os.path.join(OUR485_HIGH, f), os.path.join(OUT_VAL_HIGH, f))

    # Save manifest
    manifest = {
        "random_seed": 42,
        "total_source_pairs": len(common_files),
        "train_count": len(train_filenames),
        "val_count": len(val_filenames),
        "test_count": len(test_filenames),
        "train_filenames": train_filenames,
        "val_filenames": val_filenames,
        "test_filenames": test_filenames,
    }

    manifest_path = os.path.join(DATA_DIR, "controlled_split_manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"Manifest written to: {manifest_path}")
    print("Split creation complete!")

if __name__ == "__main__":
    create_split()

