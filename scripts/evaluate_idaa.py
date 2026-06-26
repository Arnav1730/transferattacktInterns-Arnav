import pandas as pd
import numpy as np
from pathlib import Path
from deepface import DeepFace

BASE = Path.cwd()
DATASET_ROOT = BASE / "dataset_extractedfaces" / "dataset_extractedfaces"

MODELS = [
    "Facenet512",
    "ArcFace",
    "GhostFaceNet",
    "VGG-Face"
]

pairs = pd.read_csv("outputs/ArcFace_subset_adv_paths.csv")

def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )

records = []

for model_name in MODELS:

    print(f"\n===== {model_name} =====")

    DeepFace.build_model(model_name)

    for _, row in pairs.iterrows():

        img2 = row["img2"]

        img2_local = (
            DATASET_ROOT
            / Path(img2).parent.name
            / Path(img2).name
        )

        adv_path = row["idaa_path"]

        try:

            emb_adv = DeepFace.represent(
                img_path=str(adv_path),
                model_name=model_name,
                detector_backend="skip"
            )[0]["embedding"]

            emb_target = DeepFace.represent(
                img_path=str(img2_local),
                model_name=model_name,
                detector_backend="skip"
            )[0]["embedding"]

            sim = cosine(emb_adv, emb_target)

            records.append({
                "row_id": row["row_id"],
                "attacker_model": "ArcFace",
                "victim_model": model_name,
                "dataset": row["dataset"],
                "attack_type": row["attack_type"],
                "attack_method": "IDAA",
                "variant": "vanilla",
                "similarity": sim
            })

            print(
                f"{model_name} row={row['row_id']} sim={sim:.4f}"
            )

        except Exception as e:
            print(
                f"FAILED {model_name} row={row['row_id']}"
            )
            print(e)

df = pd.DataFrame(records)
df.to_csv(
    "results_baseline/idaa_similarity_results.csv",
    index=False
)

print("\nSaved results_baseline/idaa_similarity_results.csv")
print("Rows:", len(df))
