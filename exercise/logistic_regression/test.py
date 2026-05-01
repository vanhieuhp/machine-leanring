from logistic_regression.linear_regression import LogisticRegression

X = [
    [0.1, 0.2], [0.2, 0.1], [0.1, 0.3], [0.3, 0.2], [0.2, 0.3],
    [0.8, 0.9], [0.9, 0.8], [0.7, 0.9], [0.9, 0.7], [0.8, 0.8],
    [0.3, 0.4], [0.4, 0.3], [0.6, 0.7], [0.7, 0.6], [0.5, 0.5],
]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]


# ── Chạy model từ scratch ────────────────────────────────────────

my_model = LogisticRegression(lr=0.1, epochs=1000)
my_model.fit(X, y)

print("=== Model từ scratch ===")
print(f"  a (hệ số) = {[round(v, 4) for v in my_model.a]}")
print(f"  b (intercept) = {round(my_model.b, 4)}")
print(f"  Accuracy = {my_model.score(X, y):.2%}")


from sklearn.linear_model import LogisticRegression as SklearnLR

sk_model = SklearnLR(max_iter=1000, C=1e6)
# C=1e6 nghĩa là gần như không có regularization
# để kết quả gần với model từ scratch nhất

sk_model.fit(X, y)

print("\n=== Sklearn ===")
print(f"  a (hệ số) = {[round(v, 4) for v in sk_model.coef_[0]]}")
print(f"  b (intercept) = {round(sk_model.intercept_[0], 4)}")
print(f"  Accuracy = {sk_model.score(X, y):.2%}")


# ── So sánh dự đoán từng bệnh nhân ──────────────────────────────

print("\n=== So sánh dự đoán ===")
print(f"{'BN':>4} | {'y thật':>7} | {'P (scratch)':>12} | {'P (sklearn)':>12} | {'Khớp?':>6}")
print("-" * 52)

my_probs = my_model.predict_proba(X)
sk_probs = sk_model.predict_proba(X)

for i in range(len(X)):
    my_P = my_probs[i]
    sk_P = sk_probs[i][1]   # sklearn trả về [P(class=0), P(class=1)]
    match = "✓" if abs(my_P - sk_P) < 0.05 else "~"
    print(f"{i+1:>4} | {y[i]:>7} | {my_P:>12.4f} | {sk_P:>12.4f} | {match:>6}")