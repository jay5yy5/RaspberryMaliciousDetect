import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import joblib
import tkinter as tk
from tkinter import ttk
import pathlib
feature_names = ['IAT', 'Tot sum', 'Magnitue', 'Min', 'Tot size', 'AVG', 'Max',
                 'Header_Length', 'Rate', 'Srate', 'Std', 'Radius', 'Covariance',
                 'flow_duration', 'Variance', 'Protocol Type', 'rst_count', 'urg_count',
                 'syn_count', 'HTTPS', 'ack_count', 'ack_flag_number', 'fin_count',
                 'SSH', 'DNS']
# click_method還要改，因為目前都是用stage1在測試之後要把stage2,3加進去\\\\\\\
# ===自訂義設定檔===
class WeightedVoter:
    def __init__(self, cat_path, rf_path, weights=(1, 2)):
        self.cat = CatBoostClassifier()
        self.cat.load_model(cat_path)
        self.rf = joblib.load(rf_path)
        self.w = np.array(weights, dtype=float); self.w /= self.w.sum()

        # 以 RF 的順序為基準
        self.classes_ = np.array(self.rf.classes_)

        # 若 CatBoost 類別順序不同，對齊機率欄位
        # CatBoost 沒有 classes_（若有則取用），這裡給個保守對齊寫法：
        # 假設你有存下 label encoder 或手動 all_labels，才做嚴格對齊。
        # 否則先略過（若你確認兩者順序相同）。
        self._cat_order = None  # 需要時再實作

    def predict_proba(self, X):
        p_cat = self.cat.predict_proba(X)
        p_rf  = self.rf.predict_proba(X)

        # 若需要，將 p_cat 依 self.classes_ 重新排列到一致順序
        # if self._cat_order is not None:
        #     p_cat = p_cat[:, self._cat_order]

        return self.w[0]*p_cat + self.w[1]*p_rf

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return self.classes_[idx]   # ← 回傳「類別名稱」而不是 0..K-1
# ===自訂義設定檔===
# ---- 接著再 load ----
stage1 = joblib.load('stage1_LightGBM.pkl')
p = pathlib.Path("weighted_voter_final.pkl")
stage2 = joblib.load(p)


ddos_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\ddos_mix_label99_other1.csv')
dos_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\dos_mix_label99_other1.csv')
mirai_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\mirai_mix_label99_other1.csv')
normal_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\normal_mix_label99_other1.csv')
recon_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\recon_mix_label99_other1.csv')
spoofing_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\spoofing_mix_label99_other1.csv')
bruteforce_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\bruteforce_mix_label99_other1.csv')
webbased_flow = pd.read_csv(r'C:\繼承畢業專題模型\demo\靜態流量\webbased_mix_label99_other1.csv')
# stage3 = 
root = tk.Tk()
root.title("流量預測檢視")
root.geometry("1024x1024")
main_frame = tk.Frame(root)
result_frame = tk.Frame(root)
def show_main():
    """顯示主畫面，隱藏結果畫面"""
    result_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)

def show_result():
    """顯示結果畫面，隱藏主畫面"""
    main_frame.pack_forget()
    result_frame.pack(fill="both", expand=True)

def clear_frame(frame):
    for w in frame.winfo_children():
        w.destroy()
def animate_bar(pb, target_value, duration_ms=600, steps=60):
    # 讓 ttk.Progressbar 從目前值跑到 target_value。
    # duration_ms: 動畫總時長（毫秒）
    # steps:      分成幾步前進（越多越順）
    try:
        current = float(pb["value"])
    except Exception:
        current = 0.0

    target = float(target_value)
    delta = target - current

    if steps <= 0 or duration_ms <= 0 or abs(delta) < 1e-6:
        pb["value"] = target
        return

    step_value = delta / steps
    step_delay = max(1, duration_ms // steps)

    def step(i=0):
        # 每一步把 value 加一點
        pb["value"] = float(pb["value"]) + step_value
        if i + 1 < steps:
            pb.after(step_delay, step, i + 1)
        else:
            # 最後一筆對齊到精準目標（避免累積誤差）
            pb["value"] = target

    step()

def click_method(x,pred): 
    # 兼容：如果 pred 是機率矩陣，轉為類別 argmax；若是一維就直接用
    pred = np.asarray(pred)
    non_ddos = (pred == "Non-DDoS")
    non_ddos_x = x[non_ddos]
    pred2 = stage2.predict(non_ddos_x)
    pred[non_ddos] = pred2
    print(pred2)
    if pred.ndim == 2 and pred.shape[1] > 1:
        pred_labels = pred.argmax(axis=1)
    else:
        pred_labels = pred.ravel()

    # 2) 展示label
    all_labels = ["DDoS","DoS","Normal","Mirai","Recon","Spoofing","Web-based","BruteForce"]
    s = pd.Series(pred_labels)
    counts = s.value_counts().reindex(all_labels, fill_value=0)    # 依 label 值排序
    total = int(counts.sum())

    # 3) 重畫結果畫面
    clear_frame(result_frame)

    # 標題
    tk.Label(result_frame, text="模型預測結果", font=("Arial", 16, "bold")).pack(pady=(10, 6))

    # 總覽
    tk.Label(result_frame, text=f"總筆數：{total}（只顯示實際出現的 label）", font=("Arial", 12)).pack(pady=(0, 10))

    # 進度條區塊
    bars_frame = tk.Frame(result_frame)
    bars_frame.pack(fill="x", padx=24, pady=6)

    # 用 grid 排版：每個 label 一列：名稱 | 進度條 | 百分比與數量
    row_idx = 0
    for label_value, cnt in counts.items():
        pct = (cnt / total * 100.0) if total > 0 else 0.0

        # label 名稱（字串化；若你有 mapping 可自行替換）
        name_lbl = tk.Label(bars_frame, text=str(label_value), width=18, anchor="w", font=("Arial", 11, "bold"))
        name_lbl.grid(row=row_idx, column=0, sticky="w", padx=(0, 8), pady=6)

        # 進度條
        # 進度條
        pb = ttk.Progressbar(bars_frame, orient="horizontal", length=600,
                            mode="determinate", maximum=100)
        pb.grid(row=row_idx, column=1, sticky="ew", padx=(0, 8), pady=6)
        pb["value"] = 0  # 先歸零

        # 動畫到目標百分比
        animate_bar(pb, pct, duration_ms=800, steps=100)  # 可調整速度與順暢度

        # 右側百分比+數量
        info_lbl = tk.Label(bars_frame, text=f"{pct:6.2f}%  ({cnt})", width=14, anchor="e", font=("Consolas", 11))
        info_lbl.grid(row=row_idx, column=2, sticky="e", pady=6)

        row_idx += 1

    # 若想固定顯示「全部 8 個 label」即便沒出現，可在這裡補 0% 的 bar。
    # 例如：
    # all_labels = ["BENIGN","BruteForce","DDoS","Recon","Spoofing","Web-based","X","Y"]
    # for name in all_labels:
    #     if name not in counts.index:
    #         ... 顯示 0% bar ...

    # 底部按鈕：回主畫面
    btn_frame = tk.Frame(result_frame)
    btn_frame.pack(fill="x", padx=24, pady=14)
    ttk.Button(btn_frame, text="清除 (Clear)", command=show_main).pack(side="right")

    # 切換到結果畫面
    show_result()
def ddos_on_click():
    x = ddos_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
def dos_on_click():
    x = dos_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
def mirai_on_click():
    x = mirai_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
def normal_on_click():
    x = normal_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
def recon_on_click():
    x = recon_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
def spoofing_on_click():
    x = spoofing_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
def bruteforce_on_click():
    x = bruteforce_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
def webbased_on_click():
    x = webbased_flow[feature_names[:25]]
    pred = stage1.predict(x)
    click_method(x,pred)
# ===== 主畫面內容 =====
tk.Label(main_frame, text="主畫面", font=("Arial", 16, "bold")).pack(pady=(20, 10))
style = ttk.Style()
try:
    style.theme_use("clam")
except:
    pass

def make_accent_style(name, bg, fg="white", hover=None, active=None):
    if hover is None:  hover  = bg
    if active is None: active = bg
    stylename = f"{name}.TButton"
    style.configure(stylename, foreground=fg, background=bg, padding=10, font=("Microsoft JhengHei", 11, "bold"))
    style.map(stylename,
        background=[("active", active), ("!disabled", bg)],
        foreground=[("disabled", "#aaaaaa"), ("!disabled", fg)]
    )
    return stylename

# 定義不同顏色的樣式
styles = {
    "DDOS":       make_accent_style("DDOS",       "#7b0b0b", hover="#8e0e0e", active="#5e0808"), 
    "DOS":        make_accent_style("DOS",        "#a11212", hover="#b51515", active="#861010"),
    "BENIGN":     make_accent_style("BENIGN",     "#c72a2a", hover="#d13636", active="#a82121"),
    "Mirai":      make_accent_style("Mirai",      "#db3b3b", hover="#e74c4c", active="#bf2f2f"),
    "Recon":      make_accent_style("Recon",      "#e15b5b", hover="#ea6b6b", active="#cc4d4d"),
    "Spoofing":   make_accent_style("Spoofing",   "#e97a7a", hover="#f08b8b", active="#d16666"),
    "Web-based":  make_accent_style("Web-based",  "#f29c9c", hover="#f5aaaa", active="#e27f7f"),
    "BruteForce": make_accent_style("BruteForce", "#f5b7b1", hover="#f8c9c3", active="#f2a49b"),
}

frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

def add_btn(text, style_key, cmd):
    b = ttk.Button(frame, text=text, style=styles[style_key], command=cmd)
    b.pack(fill="x", pady=6)
    return b

add_btn("DDOS", "DDOS", lambda: ddos_on_click())
add_btn("DOS", "DOS", lambda: dos_on_click())
add_btn("BENIGN", "BENIGN", lambda:normal_on_click())
add_btn("Mirai", "Mirai", lambda: mirai_on_click())
add_btn("Recon", "Recon", lambda: recon_on_click())
add_btn("Spoofing", "Spoofing", lambda: spoofing_on_click())
add_btn("Web-based", "Web-based", lambda: webbased_on_click())
add_btn("BruteForce", "BruteForce", lambda: bruteforce_on_click())
# 顯示主畫面
show_main()

root.mainloop()