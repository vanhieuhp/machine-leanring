def confusion_matrix(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP = TP + 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN = TN + 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP = FP + 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN = FN + 1

    return TP, TN, FP, FN

def accuracy(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, FP):
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)

def recall(TP, FN):
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)

# ── Chạy thử ────────────────────────────────────────

y_true = [0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 1, 0, 1, 1]  # 2 sai: FP=1, FN=1

TP, TN, FP, FN = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(f"  TP = {TP}  (dự đoán bệnh, thật bệnh)")
print(f"  TN = {TN}  (dự đoán khỏe, thật khỏe)")
print(f"  FP = {FP}  (dự đoán bệnh, thật khỏe) ← báo động giả")
print(f"  FN = {FN}  (dự đoán khỏe, thật bệnh) ← bỏ sót!")
print()
print(f"  Accuracy  = {accuracy(TP, TN, FP, FN):.2%}")
print(f"  Precision = {precision(TP, FP):.2%}")
print(f"  Recall    = {recall(TP, FN):.2%}")