import pandas as pd
from pathlib import Path

PRED_PATH = Path("folds/pred_infer.txt")
TEST_PATH = Path("test.csv")
IDS_PATH = Path("folds/infer_ids.csv")
OUT1 = Path("submission.csv")
OUT2 = Path("folds/submission.csv")


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"No existe {PRED_PATH}. Asegúrate de correr la inferencia en C++ primero.")

    preds = pd.read_csv(PRED_PATH, header=None, sep=r"\s+", engine="python")
    if preds.shape[1] < 2:
        raise ValueError(
            f"{PRED_PATH} no parece multi-clase (tiene {preds.shape[1]} columnas). "
            "¿Seguro que el modelo es 'multiclass' y no binario/regresión?"
        )

    y_hat = preds.values.argmax(axis=1).astype(int)

    if IDS_PATH.exists():
        pet_ids = pd.read_csv(IDS_PATH)["PetID"]
    else:
        if not TEST_PATH.exists():
            raise FileNotFoundError(
                f"No existe {IDS_PATH} ni {TEST_PATH}. No puedo reconstruir los PetID."
            )
        pet_ids = pd.read_csv(TEST_PATH)["PetID"]

    if len(pet_ids) != len(y_hat):
        raise ValueError(
            f"Largo mismatch: PetID={len(pet_ids)} vs preds={len(y_hat)}. "
            "¿Se generó infer.txt con el mismo test.csv?"
        )

    sub = pd.DataFrame({"PetID": pet_ids, "AdoptionSpeed": y_hat})
    sub["AdoptionSpeed"] = sub["AdoptionSpeed"].astype(int)

    sub.to_csv(OUT1, index=False)
    OUT2.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT2, index=False)

    dist = sub["AdoptionSpeed"].value_counts().sort_index()
    print("✅ submission.csv generado.")
    print(f"   -> {OUT1.resolve()}")
    print(f"   -> {OUT2.resolve()}")
    print("\nDistribución de predicciones (clase: conteo):")
    for k, v in dist.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
