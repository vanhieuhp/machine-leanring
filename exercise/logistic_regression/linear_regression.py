import math

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.a = None  # hệ số features
        self.b = 0.0  # intercept
        self.history = []

    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def _forward(self, x_list):
        # Tính z = a1*x1 + a2*x2 + ... + b
        z = self.b
        for i in range(len(self.a)):
            z = z + self.a[i] * x_list[i]

        # Tính P = sigmoid(z)
        P = self._sigmoid(z)
        return P

    def _compute_loss(self, X, y):
        n = len(y)
        total_loss = 0.0

        for i in range(n):
            P = self._forward(X[i])
            loss_i = -(y[i] * math.log(P) + (1 - y[i]) * math.log(1 - P))
            total_loss = total_loss + loss_i

        avg_loss = total_loss / n
        return avg_loss

    def fit(self, X, y):
        n_features = len(X[0])
        n = len(y)

        # Khởi tạo tất cả hệ số = 0
        self.a = []
        for i in range(n_features):
            self.a.append(0.0)
        self.b = 0.0

        # Vòng lặp gradient descent
        for epoch in range(1, self.epochs + 1):
            # ── BƯỚC 1: Forward pass ──────────────────
            # Tính P cho từng bệnh nhân
            P_list = []
            for i in range(n):
                P_i = self._forward(X[i])
                P_list.append(P_i)

            # ── BƯỚC 2: Tính gradient ─────────────────
            # grad_a[i] = trung bình của (P - y) * x_i
            # grad_b    = trung bình của (P - y)
            grad_a = []
            for i in range(n_features):
                grad_a.append(0.0)
            grad_b = 0.0

            for i in range(n):
                err = P_list[i] - y[i]  # sai số = P - y

                for j in range(n_features):
                    grad_a[j] = grad_a[j] + err * X[i][j]

                grad_b = grad_b + err

            # Chia trung bình
            for j in range(n_features):
                grad_a[j] = grad_a[j] / n
            grad_b = grad_b / n

            # ── BƯỚC 3: Cập nhật tham số ──────────────
            for j in range(n_features):
                self.a[j] = self.a[j] - self.lr * grad_a[j]
            self.b = self.b - self.lr * grad_b

            # ── Lưu loss mỗi 10 vòng ──────────────────
            if epoch % 10 == 0 or epoch == 1:
                loss = self._compute_loss(X, y)
                self.history.append((epoch, loss))
        return self

    def predict_proba(self, X):
        # Trả về xác suất P cho từng bệnh nhân
        result = []
        for i in range(len(X)):
            P = self._forward(X[i])
            result.append(P)
        return result

    def predict(self, X, threshold=0.5):
        # Trả về nhãn 0 hoặc 1
        result = []
        P_list = self.predict_proba(X)
        for i in range(len(P_list)):
            if P_list[i] >= threshold:
                result.append(1)
            else:
                result.append(0)
        return result

    def score(self, X, y):
        # Accuracy: tỉ lệ dự đoán đúng
        predictions = self.predict(X)
        n_correct = 0
        for i in range(len(y)):
            if predictions[i] == y[i]:
                n_correct = n_correct + 1
        accuracy = n_correct / len(y)
        return accuracy

    def __repr__(self):
        a_rounded = []
        for v in self.a:
            a_rounded.append(round(v, 4))
        return (f"LogisticRegression(lr={self.lr}, epochs={self.epochs})\n"
                f"  a = {a_rounded}\n"
                f"  b = {round(self.b, 4)}")