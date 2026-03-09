"""
Script 2: Fine-tune YOLO26s
============================
Fine-tune leve do YOLO26s no dataset anotado.
Usa transfer learning — só treina as últimas camadas (freeze=10).

Instalação:
  pip install ultralytics

Uso:
  python 2_finetune.py
"""

from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATASET_YAML  = "./dataset/data.yaml"
BASE_MODEL    = "yolo26s.pt"          # baixa automaticamente na primeira vez
OUTPUT_NAME   = "portrait_detector"   # nome do experimento em runs/

EPOCHS        = 30
IMG_SIZE      = 416
BATCH         = 4                     # reduz pra 4 se der OOM
FREEZE        = 20                    # congela as N primeiras camadas (transfer learning leve)
LR0           = 1e-3
DEVICE        = "cpu"                   # "cpu" se não tiver GPU

# ── TREINO ────────────────────────────────────────────────────────────────────

def main():
    model = YOLO(BASE_MODEL)

    results = model.train(
        data      = DATASET_YAML,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        freeze    = FREEZE,
        lr0       = LR0,
        device    = DEVICE,
        name      = OUTPUT_NAME,
        patience  = 10,           # early stopping
        save      = True,
        plots     = True,
        val       = True,
    )

    print(f"""
─────────────────────────────
Fine-tune concluído!
Melhor modelo: runs/detect/{OUTPUT_NAME}/weights/best.pt
Próximo passo → python 3_crop_with_model.py
─────────────────────────────
""")

if __name__ == "__main__":
    main()
