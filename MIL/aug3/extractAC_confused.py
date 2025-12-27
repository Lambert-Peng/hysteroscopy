import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# 你只要改這 2 個
# =========================
RUN_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\MIL\aug3\results\final\M1_20251227_151750_F1" # inf_M2_full_eval.py 的輸出資料夾
SOURCE_IMG_DIR = r"C:\台大碩士資料\實驗室\hysteroscopy\hysteroscopy_dataset"              # 原始圖片根目錄（bag_id 對應資料夾）
# =========================

TOP_K = 3
POOLINGS = ["max", "mean", "top3"]   # 想只做 top3 就改成 ["top3"]

# 你 inf_M2_full_eval.py 產出的檔名
PRED_XLSX = "combined_predictions.xlsx"
MIL_XLSX  = "MIL_sweep_spec.xlsx"

OUT_DIR = os.path.join(RUN_DIR, "AC_confused_from_CM")
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# -------------------------
# helpers
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def extract_bag_id(filename: str) -> str:
    parts = str(filename).split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return str(filename)

def resolve_bag_src_path(bag_id: str) -> str:
    """
    嘗試支援兩種資料夾結構：
      1) nested: root/A63/A4
      2) flat:   root/A63_A4
    """
    bag_id = str(bag_id).strip()
    nested = os.path.join(SOURCE_IMG_DIR, bag_id.replace("_", os.sep))
    if os.path.isdir(nested):
        return nested
    flat = os.path.join(SOURCE_IMG_DIR, bag_id)
    return flat

def get_indices_for_pooling(p_abn: np.ndarray, pooling: str, top_k: int = 3):
    n = len(p_abn)
    if n == 0:
        return []

    if pooling == "max":
        return [int(np.argmax(p_abn))]
    if pooling == "mean":
        return list(range(n))
    if pooling == "top3":
        order = np.argsort(-p_abn)
        k = min(top_k, n)
        return order[:k].tolist()

    raise ValueError("pooling must be max/mean/top3")

def mil_predict_one_bag(df_bag: pd.DataFrame, pooling: str, thr: float, top_k: int = 3):
    """
    跟 inf_M2_full_eval.py 同邏輯：
      Stage1: pooled p_abn vs thr -> B
      Stage2: abnormal -> sum(p_abn_i * pA_cond_i) vs sum(p_abn_i * pC_cond_i) -> A/C
    """
    p_abn = df_bag["p_abn"].to_numpy(float)
    pA_cond = df_bag["p_A_cond"].to_numpy(float)
    pC_cond = df_bag["p_C_cond"].to_numpy(float)

    I = get_indices_for_pooling(p_abn, pooling, top_k=top_k)
    if len(I) == 0:
        return 1  # fallback B

    if pooling == "max":
        S = float(np.max(p_abn))
    else:
        S = float(np.mean(p_abn[I]))

    # Stage1: A+C vs B
    if S < thr:
        return 1  # B

    # Stage2: A vs C (weighted)
    A_score = float(np.sum(p_abn[I] * pA_cond[I]))
    C_score = float(np.sum(p_abn[I] * pC_cond[I]))
    return 0 if A_score >= C_score else 2

def copy_images_for_bags(bag_ids, out_subdir):
    ensure_dir(out_subdir)
    missing = []
    copied_bags = 0

    for bag_id in tqdm(sorted(set(map(str, bag_ids)))):
        src = resolve_bag_src_path(bag_id)
        dst = os.path.join(out_subdir, bag_id)

        if os.path.isdir(src):
            ensure_dir(dst)
            copied = 0
            for f in os.listdir(src):
                if f.lower().endswith(VALID_EXT):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                    copied += 1
            if copied > 0:
                copied_bags += 1
            else:
                # 空的就清掉
                try:
                    os.rmdir(dst)
                except:
                    pass
        else:
            missing.append(bag_id)

    return copied_bags, missing


# -------------------------
# main
# -------------------------
def main():
    ensure_dir(OUT_DIR)

    pred_path = os.path.join(RUN_DIR, PRED_XLSX)
    mil_path  = os.path.join(RUN_DIR, MIL_XLSX)

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Not found: {pred_path}")
    if not os.path.exists(mil_path):
        raise FileNotFoundError(f"Not found: {mil_path}")
    if not os.path.isdir(SOURCE_IMG_DIR):
        raise FileNotFoundError(f"SOURCE_IMG_DIR not found: {SOURCE_IMG_DIR}")

    # 1) 讀 instance-level combined predictions
    df = pd.read_excel(pred_path)
    required = {"filename", "true_label", "p_abn", "p_A_cond", "p_C_cond"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {PRED_XLSX}: {missing}")

    if "bag_id" not in df.columns:
        df["bag_id"] = df["filename"].apply(extract_bag_id)

    # bag true label（眾數）
    bag_true = df.groupby("bag_id")["true_label"].agg(lambda x: int(x.value_counts().idxmax()))

    # 2) 讀 MIL 報告，拿每種 pooling 的 best_T_abn
    summary = pd.read_excel(mil_path, sheet_name="summary")
    if not {"pooling", "best_T_abn"}.issubset(summary.columns):
        raise ValueError("MIL summary sheet must contain columns: pooling, best_T_abn")

    thr_map = {r["pooling"]: float(r["best_T_abn"]) for _, r in summary.iterrows()}

    # 3) 對每個 pooling：重建 bag-level pred -> 找 A<->C confusion -> 輸出清單 + 複製圖片
    out_excel = os.path.join(OUT_DIR, "AC_confusion_from_CM.xlsx")
    writer = pd.ExcelWriter(out_excel, engine="xlsxwriter")

    for pooling in POOLINGS:
        if pooling not in thr_map:
            print(f"[WARN] pooling={pooling} not in MIL summary, skip.")
            continue

        thr = thr_map[pooling]
        print(f"\n[INFO] pooling={pooling}, best_T_abn={thr:.4f}")

        bag_rows = []
        for bag_id, g in df.groupby("bag_id"):
            pred = mil_predict_one_bag(g, pooling=pooling, thr=thr, top_k=TOP_K)
            bag_rows.append({
                "bag_id": bag_id,
                "true_label": int(bag_true.loc[bag_id]),
                "pred_label": int(pred),
            })

        df_bag = pd.DataFrame(bag_rows)

        # A->C and C->A
        df_A_to_C = df_bag[(df_bag["true_label"] == 0) & (df_bag["pred_label"] == 2)].copy()
        df_C_to_A = df_bag[(df_bag["true_label"] == 2) & (df_bag["pred_label"] == 0)].copy()

        print(f"  True A -> Pred C : {len(df_A_to_C)} bags")
        print(f"  True C -> Pred A : {len(df_C_to_A)} bags")

        # instance list for those bags
        bags_A_to_C = set(df_A_to_C["bag_id"].astype(str))
        bags_C_to_A = set(df_C_to_A["bag_id"].astype(str))

        inst_cols = ["bag_id", "filename", "true_label", "p_abn", "p_A_cond", "p_C_cond"]
        inst_A_to_C = df[df["bag_id"].astype(str).isin(bags_A_to_C)][inst_cols].copy()
        inst_C_to_A = df[df["bag_id"].astype(str).isin(bags_C_to_A)][inst_cols].copy()

        # write excel sheets
        df_bag.to_excel(writer, sheet_name=f"{pooling}_bags"[:31], index=False)
        df_A_to_C.to_excel(writer, sheet_name=f"{pooling}_AtoC_bags"[:31], index=False)
        df_C_to_A.to_excel(writer, sheet_name=f"{pooling}_CtoA_bags"[:31], index=False)
        inst_A_to_C.to_excel(writer, sheet_name=f"{pooling}_AtoC_inst"[:31], index=False)
        inst_C_to_A.to_excel(writer, sheet_name=f"{pooling}_CtoA_inst"[:31], index=False)

        # copy images
        img_out = os.path.join(OUT_DIR, f"Images_{pooling}_bestT_{thr:.3f}")
        ensure_dir(img_out)

        if bags_A_to_C:
            sub = os.path.join(img_out, f"TrueA_PredC_Top{TOP_K}")
            copied, miss = copy_images_for_bags(bags_A_to_C, sub)
            print(f"  [COPY] A->C copied_bags={copied}, missing={len(miss)}")
            if miss:
                pd.DataFrame({"missing_bag_id": miss}).to_csv(os.path.join(sub, "missing_bags.csv"), index=False)

        if bags_C_to_A:
            sub = os.path.join(img_out, f"TrueC_PredA_Top{TOP_K}")
            copied, miss = copy_images_for_bags(bags_C_to_A, sub)
            print(f"  [COPY] C->A copied_bags={copied}, missing={len(miss)}")
            if miss:
                pd.DataFrame({"missing_bag_id": miss}).to_csv(os.path.join(sub, "missing_bags.csv"), index=False)

    writer.close()
    print("\n" + "=" * 70)
    print("[DONE]")
    print(f"Saved lists: {out_excel}")
    print(f"Saved images under: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()