# RNN Sin Wave Prediction — Kế hoạch chi tiết

## Tổng quan

Bài toán: học dự đoán sóng sin bằng RNN — kiến trúc cơ bản nhất để hiểu sequence modeling.

```
Input:  [x_t, x_{t+1}, ..., x_{t+9}]   (10 bước)
Output: x_{t+10}                         (1 bước tiếp theo)
```

---

## Yêu cầu kỹ thuật

| Hạng mục       | Chi tiết                                          |
|----------------|---------------------------------------------------|
| Data           | `torch.sin(torch.arange(0, 100, 0.1))` — 1000 pt |
| Window size    | 10 bước nhìn lại (look-back)                      |
| Prediction     | 1 bước tiếp theo (next-step)                      |
| Model 1        | RNN scratch (`nn.Module` + `nn.Parameter`)        |
| Model 2        | `nn.RNN` (verification)                           |
| Loss target    | MSE < 0.01                                        |
| Visualization  | Predicted vs Actual phải bám sát nhau             |

---

## Step-by-Step Plan

### Step 1 — Data Generation & EDA

```python
data = torch.sin(torch.arange(0, 100, 0.1))   # 1000 điểm, range [-1, 1]
```

- Vẽ toàn bộ signal để kiểm tra chu kỳ
- Sin đã normalize trong `[-1, 1]` → không cần scale thêm

**Output:** tensor shape `(1000,)`

---

### Step 2 — Tạo Sequences (Sliding Window)

```
data[0:10]  → target: data[10]
data[1:11]  → target: data[11]
...
data[989:999] → target: data[999]
```

- Tổng: **990 samples**
- X shape: `(990, 10, 1)` — batch × seq_len × features
- y shape: `(990, 1)`
- Train/Test split: 80/20 → **Train: 792, Test: 198**

---

### Step 3 — RNN Scratch Implementation

Công thức tổng quát của RNN cell:

```
h_t = tanh(x_t · W_xh + h_{t-1} · W_hh + b_h)
ŷ   = h_T · W_hy + b_y          (dùng hidden state cuối)
```

Các parameters cần học:

| Parameter | Shape       | Vai trò                 |
|-----------|-------------|-------------------------|
| `W_xh`    | `(1, H)`    | input → hidden          |
| `W_hh`    | `(H, H)`    | hidden → hidden         |
| `b_h`     | `(H,)`      | bias hidden             |
| `W_hy`    | `(H, 1)`    | hidden → output         |
| `b_y`     | `(1,)`      | bias output             |

`H = hidden_size = 64`

```python
class RNNScratch(nn.Module):
    def forward(self, x):
        h = zeros(batch, H)
        for t in range(seq_len):
            h = tanh(x[:,t,:] @ W_xh + h @ W_hh + b_h)
        return h @ W_hy + b_y
```

---

### Step 4 — Training Loop

```
Optimizer : Adam (lr = 0.005)
Loss      : MSELoss
Epochs    : 500
Batch     : full batch (792 samples)
```

Tại sao Adam thay vì SGD?
- Adam có adaptive learning rate → hội tụ nhanh hơn với dữ liệu sin
- SGD cần nhiều epoch hơn và tuning lr kỹ hơn

**Checkpoint:** in loss mỗi 50 epoch, verify `loss < 0.01` sau epoch cuối.

---

### Step 5 — Visualization

**Plot 1 — Loss Curve:**
```
x-axis: epoch
y-axis: MSE (log scale)
line:   đường đỏ nét đứt tại y=0.01 (target threshold)
```

**Plot 2 — Predicted vs Actual (Test Set):**
```
blue solid : actual values
red dashed : predicted values
```

Predicted phải bám sát actual với sai số không rõ ràng bằng mắt thường.

---

### Step 6 — Verify với `nn.RNN`

```python
class RNNModel(nn.Module):
    def __init__(self):
        self.rnn = nn.RNN(1, 64, batch_first=True)
        self.fc  = nn.Linear(64, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])   # lấy hidden state cuối
```

- Cùng hyperparameters với scratch version
- So sánh loss curve và predictions
- Kết quả phải tương đương → xác nhận scratch đúng

---

### Step 7 — Comparison Dashboard

```
┌─────────────────────┬──────────────────────────┐
│  Loss Comparison    │  Predicted vs Actual      │
│  (log scale)        │  (Test Set)               │
│                     │                           │
│  Scratch ─────      │  Actual ────              │
│  nn.RNN  ─────      │  Scratch - - -            │
│  Target  ·····      │  nn.RNN  ·····            │
└─────────────────────┴──────────────────────────┘
```

---

## Kiến trúc RNN (Minh họa)

```
Thời gian →   t=0    t=1    ...  t=9
                │      │           │
x_t ──────────►[cell]─[cell]─...─[cell]──► h_9
               ↑       ↑              │
               h_{t-1} h_{t-1}        │
                                      ▼
                                   [fc layer]
                                      │
                                      ▼
                                    ŷ (prediction)
```

---

## Verification Checklist

```
[ ] Data shape: X=(990,10,1), y=(990,1)
[ ] Scratch RNN: tất cả params là nn.Parameter (requires_grad=True)
[ ] Hidden state reset về zeros mỗi forward pass
[ ] Training loss < 0.01 sau 500 epochs
[ ] Test plot: predicted bám sát actual (không lệch pha)
[ ] nn.RNN loss tương đương scratch loss (±20%)
```

---

## Troubleshooting

| Vấn đề                  | Nguyên nhân              | Giải pháp                        |
|-------------------------|--------------------------|----------------------------------|
| Loss không giảm         | lr quá lớn/nhỏ           | Thử lr=0.001 hoặc lr=0.01        |
| Loss ổn định > 0.01     | Hidden size nhỏ          | Tăng hidden_size lên 128         |
| Prediction lệch pha     | Seq_len không phù hợp    | Tăng seq_len lên 20              |
| Gradient exploding      | W_hh khởi tạo sai        | Dùng orthogonal init             |

---

## File thực hiện

```
rnn/scratch_sin.ipynb   ← notebook đầy đủ (scratch + nn.RNN + plots)
```