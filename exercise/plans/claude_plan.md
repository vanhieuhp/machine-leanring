# 📚 Hành Trình Học Machine Learning — Eric Nguyen Van

> Cập nhật: Tháng 5/2026 | Version 2.0

---

## 🗺️ Tổng quan lộ trình (5 Phases + 2 Tracks)

| Phase | Tên | Trạng thái | Thời gian ước tính |
|-------|-----|------------|--------------------|
| Phase 0 | Toán nền tảng | ✅ Hoàn thành | — |
| Phase 1 | Classical ML | ✅ Hoàn thành | — |
| Phase 2 | Neural Networks từ scratch | 🔄 Tiếp theo | ~6–8 tuần |
| Phase 3 | Deep Learning Architectures | ⏳ Chưa bắt đầu | ~8–10 tuần |
| Phase 4 | LLMs & Modern AI | ⏳ Chưa bắt đầu | ~6–8 tuần |
| Phase 5 | MLOps & Production | ⏳ Chưa bắt đầu | ~4–6 tuần |
| Track CV | Computer Vision (Specialization) | ⏳ Tùy chọn — sau Phase 4 | — |
| Track VLM | Multimodal AI — Vision + Language | ⏳ Tùy chọn — sau Phase 4 + Track CV | — |

---

## ✅ Phase 0 — Toán Nền Tảng (Hoàn thành)

### Linear Algebra
- Vectors và các phép toán cơ bản
- Dot product (tích vô hướng)
- Matrix multiplication (nhân ma trận)
- Attention mechanism (giới thiệu sớm)
- Norms (L1, L2)

### Probability & Statistics
- Các khái niệm xác suất nền tảng
- Thống kê cơ bản

---

## ✅ Phase 1 — Classical ML (Hoàn thành)

### 1. Linear Regression ✅
- Derive Normal Equation từ first principles
- MSE loss và gradient descent
- Closed-form vs Gradient Descent
- **Milestone project:** Dự đoán giá nhà

### 2. Overfitting & Validation ✅
- Khái niệm overfitting / underfitting
- Train/Validation split
- Xác nhận overfitting qua train vs validation RMSE

### 3. Logistic Regression ✅ *(học sâu nhất)*
- Sigmoid derive, đạo hàm `dP/dz = P(1-P)`
- Binary Cross-Entropy từ MLE
- Gradient = `P - y` (full backward pass)
- Class hoàn chỉnh + Accuracy, Precision, Recall, ROC, AUC
- Verify với sklearn trên Breast Cancer Wisconsin

### 4. Decision Tree ✅
- Entropy từ information theory
- Information Gain derive

### 5. Random Forest ✅
- Bias-variance tradeoff, bootstrapping

### 6. SVM ✅
- Margin maximization, QP problem, Lagrangian, kernel trick

### 7. K-Means ✅ *(cần bổ sung bản chất)*
- Đã code được — chưa derive Inertia, chưa hiểu convergence

### 8. PCA ✅ *(cần bổ sung bản chất — ưu tiên cao)*
- Chỉ mới dùng sklearn — CHƯA học covariance matrix, eigenvector, SVD

### 🏆 Milestone Project — Phase 1
- **Dataset:** Heart Disease
- **Thuật toán:** Logistic Regression
- **Kết quả:** Accuracy **86.67%**, Recall **82.61%**

---

## 🔄 Phase 2 — Neural Networks từ Scratch

> **Triết lý:** Hiểu từng electron trước khi lắp bóng đèn.
> Toàn bộ phase này sẽ xây bằng NumPy thuần — không dùng PyTorch, không dùng TensorFlow.

---

### Mục tiêu cuối phase
Xây được một **Neural Network nhiều lớp từ zero**, train được trên ảnh số viết tay (MNIST),
hiểu từng dòng code, từng con số trong quá trình forward và backward pass.

---

### 2.1 — Một Neuron Là Gì?

**Câu hỏi mở đầu:** *"Logistic Regression mà mình đã học — thực ra đó chính là 1 neuron. Tại sao?"*

**Cần hiểu:**
- Một neuron = linear combination + activation function
- `z = w₁x₁ + w₂x₂ + ... + b` (giống hệt Logistic Regression)
- `output = sigmoid(z)`
- Sự khác biệt duy nhất: khi ta **xếp nhiều neuron lại**, ta có "layer"

**Toán học:**
- Review dot product: `z = W · x + b`
- Tại sao cần bias `b`? (tương tự hệ số `b` trong `y = ax + b`)

**Milestone nhỏ:** Vẽ được sơ đồ 1 neuron, giải thích được từng thành phần

---

### 2.2 — Tại Sao Cần Nhiều Lớp? (Non-linearity)

**Câu hỏi:** *"Nếu ta xếp 100 neuron tuyến tính chồng nhau — kết quả có khác gì 1 neuron không?"*

**Cần hiểu (toán):**
- Nhiều linear layers = vẫn là 1 linear function → vô dụng
- Bằng chứng: `W₂(W₁x) = (W₂W₁)x = W_combined · x`
- Non-linearity = khả năng học đường cong, không gian phức tạp

**Các activation function — derive + so sánh:**

| Function | Công thức | Đạo hàm | Khi nào dùng |
|----------|-----------|---------|--------------|
| Sigmoid | `1/(1+e⁻ᶻ)` | `σ(z)(1-σ(z))` | Output binary |
| Tanh | `(eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | `1 - tanh²(z)` | Hidden layers (cũ) |
| ReLU | `max(0, z)` | `0 nếu z<0, 1 nếu z>0` | Hidden layers (phổ biến nhất) |
| Leaky ReLU | `max(0.01z, z)` | `0.01 nếu z<0, 1 nếu z>0` | Tránh "dying ReLU" |
| Softmax | `eᶻⁱ / Σeᶻʲ` | (ma trận Jacobian) | Output multi-class |

**Tại sao ReLU thắng Sigmoid ở hidden layers?**
- Vanishing gradient problem: gradient sigmoid → 0 khi `|z|` lớn
- ReLU: gradient luôn = 1 hoặc 0 (không vanish)
- Tính toán đơn giản hơn → train nhanh hơn

---

### 2.3 — Forward Pass (Lan Truyền Xuôi)

**Cấu trúc mạng đơn giản (2 lớp):**
```
Input x → [Layer 1: W₁, b₁, ReLU] → [Layer 2: W₂, b₂, Softmax] → Output ŷ
```

**Toán học từng bước:**
```
Layer 1:
  z₁ = W₁ · x + b₁          (linear transformation)
  a₁ = ReLU(z₁)              (activation)

Layer 2:
  z₂ = W₂ · a₁ + b₂         (linear transformation)
  a₂ = Softmax(z₂)           (activation → xác suất)

Output:
  ŷ = a₂                     (vector xác suất các class)
```

**Dimension tracking (quan trọng!):**
- Input: `(n_features, 1)` hoặc `(n_features, batch_size)`
- W₁: `(n_hidden, n_features)`
- z₁ = W₁x + b₁: `(n_hidden, 1)`
- W₂: `(n_output, n_hidden)`
- z₂: `(n_output, 1)`

**Code sẽ viết:**
```python
def forward(x, W1, b1, W2, b2):
    z1 = 0
    for i in range(len(W1)):
        for j in range(len(x)):
            z1[i] = z1[i] + W1[i][j] * x[j]
        z1[i] = z1[i] + b1[i]
    a1 = relu(z1)
    # ... tiếp tục
```

---

### 2.4 — Loss Function cho Multi-class

**Câu hỏi:** *"Binary Cross-Entropy ta đã biết. Multi-class thì sao?"*

**Categorical Cross-Entropy — derive từ MLE:**
```
Mục tiêu: maximize P(y|x) với y là one-hot vector
P(y|x) = ∏ ŷᵢ^yᵢ    (chỉ class đúng mới có yᵢ=1)
Log-likelihood: log L = ∑ yᵢ log(ŷᵢ)
Loss = -∑ yᵢ log(ŷᵢ)  (negative vì ta minimize)
```

**Ví dụ cụ thể:**
```
y = [0, 1, 0]  (đúng là class 1)
ŷ = [0.1, 0.7, 0.2]  (model dự đoán)
Loss = -(0*log(0.1) + 1*log(0.7) + 0*log(0.2)) = -log(0.7) ≈ 0.357
```

---

### 2.5 — Backpropagation (Trái Tim của Deep Learning)

> Đây là phần **quan trọng nhất và khó nhất** của Phase 2.
> Dành nhiều thời gian nhất ở đây.

**Bước chuẩn bị — Chain Rule:**
```
Nếu f(g(x)), thì df/dx = df/dg × dg/dx
```

**Ví dụ trực giác:** Nếu lương tăng → thuế tăng → tài sản thay đổi.
Tài sản thay đổi bao nhiêu khi lương tăng 1%? = (rate thuế thay đổi) × (tài sản/thuế)

**Computational Graph:**
```
x → [z = Wx+b] → [a = ReLU(z)] → [L = Loss(a, y)]

Backward:
dL/dz = dL/da × da/dz   (chain rule)
dL/dW = dL/dz × dz/dW   (chain rule tiếp)
dL/db = dL/dz × dz/db
```

**Backward pass đầy đủ (2-layer network):**
```
Bước 1: dL/dz₂ = ŷ - y         (softmax + cross-entropy kết hợp đẹp!)
Bước 2: dL/dW₂ = dL/dz₂ · a₁ᵀ
Bước 3: dL/db₂ = dL/dz₂
Bước 4: dL/da₁ = W₂ᵀ · dL/dz₂
Bước 5: dL/dz₁ = dL/da₁ × ReLU'(z₁)
Bước 6: dL/dW₁ = dL/dz₁ · xᵀ
Bước 7: dL/db₁ = dL/dz₁
```

**Gradient Descent update:**
```
W₁ = W₁ - α × dL/dW₁
b₁ = b₁ - α × dL/db₁
W₂ = W₂ - α × dL/dW₂
b₂ = b₂ - α × dL/db₂
```

**Code sẽ viết:**
```python
def backward(x, y, z1, a1, z2, a2, W1, W2):
    dz2 = a2 - y          # softmax + cross-entropy
    dW2 = []
    for i in range(len(dz2)):
        row = []
        for j in range(len(a1)):
            row.append(dz2[i] * a1[j])
        dW2.append(row)
    # ... tiếp tục
```

---

### 2.6 — Vanishing Gradient Problem

**Câu hỏi:** *"Điều gì xảy ra khi network có 10 lớp và ta dùng sigmoid ở mỗi lớp?"*

**Vấn đề:**
- Đạo hàm sigmoid tối đa = 0.25 (tại z=0)
- Qua 10 lớp: 0.25¹⁰ ≈ 0.0000001 → gradient gần như = 0
- Lớp đầu không học được gì → network không train được

**Giải pháp:**
- Dùng ReLU thay sigmoid ở hidden layers
- Batch Normalization (học ở Phase 3)
- Residual connections / Skip connections (học ở Phase 3)

---

### 2.7 — Optimization: Vượt Ra Ngoài SGD Đơn Giản

**SGD (Stochastic Gradient Descent) — đã biết:**
```
w = w - α × ∇L
```

**Vấn đề của SGD thuần:**
- Bước nhảy quá cứng nhắc — không thích nghi
- Dao động mạnh ở những vùng có gradient lớn
- Chậm ở những vùng gradient nhỏ

**Momentum — thêm "quán tính":**
```
v = β × v + (1-β) × ∇L     (velocity = trung bình trọng số của gradient)
w = w - α × v
```
*Ý tưởng: như một quả bóng lăn dốc — tích lũy tốc độ theo hướng đúng*

**RMSprop — thích nghi learning rate:**
```
s = β × s + (1-β) × (∇L)²  (moving average của gradient²)
w = w - (α / √s) × ∇L
```
*Ý tưởng: feature nào có gradient lớn → giảm learning rate; ngược lại tăng lên*

**Adam — kết hợp cả hai (phổ biến nhất hiện tại):**
```
m = β₁ × m + (1-β₁) × ∇L           (momentum)
v = β₂ × v + (1-β₂) × (∇L)²        (RMSprop)
m̂ = m / (1-β₁ᵗ)                    (bias correction)
v̂ = v / (1-β₂ᵗ)                    (bias correction)
w = w - α × m̂ / (√v̂ + ε)
```
*Mặc định: β₁=0.9, β₂=0.999, ε=1e-8*

---

### 2.8 — Weight Initialization

**Câu hỏi:** *"Khởi tạo W = 0 có vấn đề gì không?"*

**Vấn đề symmetry breaking:**
- Nếu W=0: mọi neuron tính cùng một giá trị → mọi gradient giống nhau → không học được
- Cần phá vỡ symmetry bằng random init

**Các phương pháp:**
- Random init thông thường: `W ~ N(0, 0.01)` → quá nhỏ → vanishing
- **Xavier / Glorot init** (cho tanh): `W ~ N(0, √(2/(n_in + n_out)))`
- **He init** (cho ReLU): `W ~ N(0, √(2/n_in))` — phổ biến nhất hiện nay

---

### 2.9 — Mini-batch Gradient Descent

**3 chế độ training:**
| Chế độ | Batch size | Ưu điểm | Nhược điểm |
|--------|-----------|---------|-----------|
| Batch GD | Toàn bộ dataset | Stable convergence | Chậm, tốn RAM |
| SGD | 1 sample | Nhanh update | Noisy, dao động |
| Mini-batch GD | 32–256 | Cân bằng tốt | Cần chọn batch size |

**Tại sao mini-batch thắng?**
- GPU hoạt động hiệu quả nhất với matrix operations
- Một batch 64 samples = một ma trận 64 cột → tính song song
- Noise vừa phải giúp thoát local minima

---

### 2.10 — Regularization trong Neural Networks

**L2 Regularization (Weight Decay):**
```
Loss_total = Loss_original + λ × Σ w²
→ Gradient: ∂Loss/∂w = ∂Loss_original/∂w + 2λw
→ Update:   w = w(1 - 2αλ) - α × ∂Loss_original/∂w
```

**Dropout:**
- Trong training: ngẫu nhiên "tắt" p% neuron mỗi batch
- Trong inference: dùng toàn bộ neuron, nhân output với (1-p)
- Tại sao hoạt động: buộc network học features độc lập, không phụ thuộc lẫn nhau
- Học sâu hơn ở Phase 3

---

### 🏆 Milestone Project — Phase 2
**Xây Neural Network phân loại chữ số viết tay (MNIST)**
- Input: ảnh 28×28 pixel = 784 features
- Architecture: `784 → 128 → 64 → 10`
- Activation: ReLU (hidden), Softmax (output)
- Loss: Categorical Cross-Entropy
- Optimizer: Adam
- **Mục tiêu:** Accuracy > 97% — viết 100% bằng NumPy, không framework

---

## ⏳ Phase 3 — Deep Learning Architectures

> **Triết lý:** Mỗi kiến trúc ra đời để giải quyết 1 vấn đề cụ thể.
> Luôn hỏi: *"Vấn đề gì khiến người ta phát minh ra cái này?"*

---

### Mục tiêu cuối phase
Hiểu và implement được CNN cho ảnh, RNN/LSTM cho chuỗi,
biết khi nào dùng kiến trúc nào và tại sao.

---

### 3.1 — Convolutional Neural Networks (CNN)

#### 3.1.1 Vấn đề của Fully Connected Network với ảnh

**Câu hỏi:** *"Ảnh 224×224 màu RGB có bao nhiêu pixel? Nếu layer đầu có 1000 neuron, cần bao nhiêu tham số?"*

Tính: 224 × 224 × 3 = 150,528 input → × 1000 neuron = **150 triệu tham số** chỉ ở layer đầu!

**Vấn đề:**
- Quá nhiều tham số → overfit, chậm
- Không tận dụng được cấu trúc không gian của ảnh (pixel gần nhau mới liên quan)
- Không translation invariant (con mèo ở góc trái ≠ con mèo ở góc phải với FC network)

#### 3.1.2 Convolution Operation — Từ Đầu

**Ý tưởng:** Thay vì kết nối mọi pixel với mọi neuron, chỉ kết nối **vùng cục bộ** (local region)

**Convolution thủ công:**
```
Input ảnh (5×5):          Filter/Kernel (3×3):
1 2 3 4 5                 1 0 -1
6 7 8 9 10    →  ×        1 0 -1    →  Feature Map (3×3)
...                       1 0 -1
```

Mỗi vị trí trong feature map = dot product của filter với vùng tương ứng trên ảnh

**Các khái niệm cần derive:**
- **Stride:** Bước nhảy của filter (stride=1 vs stride=2)
- **Padding:** Thêm viền 0 để giữ kích thước (`same` vs `valid`)
- **Output size formula:** `(W - F + 2P) / S + 1`
  - W = input width, F = filter size, P = padding, S = stride

**Tại sao filter học được feature?**
- Filter phát hiện cạnh dọc: `[[1,0,-1],[1,0,-1],[1,0,-1]]`
- Filter phát hiện cạnh ngang: `[[1,1,1],[0,0,0],[-1,-1,-1]]`
- Trong training, filter được **học tự động** qua backprop

#### 3.1.3 Parameter Sharing — Vì Sao CNN Hiệu Quả

**Key insight:** Cùng 1 filter được dùng trên TOÀN BỘ ảnh
- Phát hiện cạnh dọc ở góc trái = phát hiện cạnh dọc ở góc phải = cùng filter
- 3×3 filter = 9 tham số (thay vì 150 triệu)

**Số tham số trong Conv layer:**
```
Params = F × F × C_in × C_out + C_out (bias)
F = filter size, C_in = channels vào, C_out = số filters
```
Ví dụ: 32 filters 3×3 trên ảnh RGB → 3×3×3×32 + 32 = **896 tham số**

#### 3.1.4 Pooling Layers

**Max Pooling (phổ biến nhất):**
```
Input (4×4):         Max Pool 2×2, stride 2:
1 3 2 4              3 4
5 6 1 2    →         6 3
3 2 4 1
1 0 3 2
```

**Tại sao cần pooling?**
- Giảm kích thước không gian → giảm tính toán
- Translation invariance: dịch chuyển nhỏ không ảnh hưởng đến output
- Tăng receptive field (vùng ảnh mà một neuron "nhìn thấy")

#### 3.1.5 Kiến Trúc CNN Kinh Điển

**LeNet-5 (1998) — CNN đầu tiên thực dụng:**
```
Input(32×32) → Conv(5×5,6) → Pool → Conv(5×5,16) → Pool → FC(120) → FC(84) → Output(10)
```

**AlexNet (2012) — cuộc cách mạng:**
- Đánh bại các phương pháp truyền thống ở ImageNet competition
- Lần đầu dùng ReLU, Dropout, GPU training ở scale lớn

**VGG (2014):**
- Ý tưởng: chỉ dùng filter 3×3, stack nhiều lớp
- Tại sao 3×3 thắng? Hai lớp 3×3 = receptive field 5×5 nhưng ít tham số hơn

**ResNet (2015) — Residual Connections:**
```
Output = F(x) + x    (skip connection)
```
- Giải quyết vanishing gradient cho network rất sâu (50-152 lớp)
- Nếu F(x) = 0 → output = x → layer "học identity" → không làm hại
- Ý tưởng: học residual dễ hơn học toàn bộ mapping

---

### 3.2 — Recurrent Neural Networks (RNN)

#### 3.2.1 Vấn đề với Dữ Liệu Chuỗi

**Câu hỏi:** *"Feed-forward network xử lý 'Tôi thích ăn phở' và 'Tôi không thích ăn phở' — có phân biệt được 'không' ở giữa ảnh hưởng đến nghĩa cả câu không?"*

**Vấn đề:** FC network và CNN không có "bộ nhớ" — mỗi input được xử lý độc lập

**Ứng dụng cần bộ nhớ:**
- Dịch thuật: "bank" (ngân hàng hay bờ sông?) phụ thuộc context
- Dự báo thời tiết: ngày hôm nay phụ thuộc ngày hôm qua
- Speech recognition: âm phụ thuộc âm trước

#### 3.2.2 RNN — Cơ Chế Hidden State

**Công thức:**
```
hₜ = tanh(Wₕ × hₜ₋₁ + Wₓ × xₜ + b)
yₜ = Wᵧ × hₜ + bᵧ
```

- `hₜ` = hidden state tại thời điểm t (bộ nhớ)
- `xₜ` = input tại thời điểm t
- `hₜ₋₁` = hidden state từ bước trước

**Key insight:** Cùng một bộ trọng số W được dùng ở **mọi bước thời gian** (parameter sharing theo time)

#### 3.2.3 BPTT — Backpropagation Through Time

**Vấn đề:** Gradient phải lan truyền ngược qua nhiều bước thời gian

**Vanishing gradient trong RNN:**
- Với chuỗi dài 100 từ: gradient từ từ 100 phải đi ngược về từ 1
- Qua 100 bước nhân: `0.9^100 ≈ 0.000027` → quên hoàn toàn

**Exploding gradient:**
- Ngược lại: gradient tăng theo hàm mũ → NaN
- Giải pháp: Gradient clipping (cắt gradient nếu > ngưỡng)

#### 3.2.4 LSTM — Long Short-Term Memory

**Ý tưởng:** Thêm cơ chế "cổng" để kiểm soát thông tin nào được nhớ, quên, output

**4 cổng của LSTM:**
```
Forget gate:  fₜ = σ(Wf × [hₜ₋₁, xₜ] + bf)   → quên bao nhiêu từ cell state cũ
Input gate:   iₜ = σ(Wi × [hₜ₋₁, xₜ] + bi)    → cập nhật bao nhiêu thông tin mới
Cell gate:    c̃ₜ = tanh(Wc × [hₜ₋₁, xₜ] + bc) → thông tin mới cần cập nhật
Output gate:  oₜ = σ(Wo × [hₜ₋₁, xₜ] + bo)   → output bao nhiêu từ cell state

Cell state:   Cₜ = fₜ × Cₜ₋₁ + iₜ × c̃ₜ
Hidden state: hₜ = oₜ × tanh(Cₜ)
```

**GRU — Gated Recurrent Unit (đơn giản hơn LSTM, ít tham số hơn):**
- Chỉ có 2 cổng: Reset gate và Update gate
- Thực tế performance tương đương LSTM

---

### 3.3 — Batch Normalization

**Vấn đề:** Internal Covariate Shift
- Mỗi mini-batch có distribution khác nhau
- Lớp sau liên tục phải adapt với distribution thay đổi → train chậm

**Công thức:**
```
x̂ = (x - μ_batch) / √(σ²_batch + ε)    (normalize về mean=0, std=1)
y = γ × x̂ + β                          (scale và shift có thể học)
```

- `γ` và `β`: các tham số **được học** (không phải hyperparameter)
- Cho phép network tự quyết định distribution nào là tốt nhất

**Tác dụng:**
- Train nhanh hơn (learning rate cao hơn)
- Ít nhạy cảm với weight initialization
- Có tác dụng regularization nhẹ

---

### 3.4 — Dropout (Học Sâu Hơn)

**Lý thuyết ensemble:** Dropout ≈ train và average nhiều subnetwork khác nhau

**Inverted Dropout (implementation chuẩn):**
```python
mask = []
for i in range(len(a)):
    if random() > dropout_rate:
        mask.append(1.0 / (1 - dropout_rate))  # scale up
    else:
        mask.append(0)
output[i] = a[i] * mask[i]
```
- Trong inference: không dropout, không cần scale (đã bù khi train)

**Khi nào dùng Dropout?**
- Fully connected layers: thường dùng rate 0.5
- Convolutional layers: thường rate nhỏ hơn (0.1–0.2) hoặc không dùng
- Không dùng trong inference

---

### 🏆 Milestone Project — Phase 3
**Phân loại ảnh chó vs mèo bằng CNN (Transfer Learning)**
- Dùng pretrained ResNet50 (ImageNet weights)
- Fine-tune trên dataset nhỏ (~2000 ảnh)
- Hiểu tại sao transfer learning hoạt động
- So sánh: train từ đầu vs fine-tune
- **Mục tiêu:** Accuracy > 90% với dataset nhỏ

---

## ⏳ Phase 4 — LLMs & Modern AI

> **Triết lý:** Hiểu tại sao Transformer thay thế RNN,
> rồi hiểu tại sao GPT và BERT là hai hướng khác nhau từ cùng 1 kiến trúc.

---

### Mục tiêu cuối phase
Hiểu Transformer từ bên trong, tự implement Attention từ scratch,
hiểu GPT và BERT hoạt động ra sao, và biết cách fine-tune LLM cho task cụ thể.

---

### 4.1 — Vấn Đề Của RNN (Nhìn Lại)

**3 vấn đề lớn của RNN/LSTM:**
1. **Sequential computation:** Bước t phải đợi bước t-1 → không song song hóa được → chậm với chuỗi dài
2. **Long-range dependency:** Dù có LSTM, context rất xa vẫn bị suy giảm
3. **Bottleneck:** Toàn bộ thông tin chuỗi dài phải nén vào 1 vector hidden state

**Câu hỏi dẫn dắt:** *"Khi bạn đọc sách và cần tham chiếu đến trang 1 từ trang 100, bạn làm gì? Bạn không cố nhớ mọi thứ — bạn quay lại trang 1 để đọc. Attention làm điều tương tự."*

---

### 4.2 — Attention Mechanism

#### 4.2.1 Intuition — Database Analogy

**Ý tưởng:** Mỗi từ trong chuỗi có thể "hỏi" tất cả các từ khác: *"Từ nào liên quan đến tôi nhất?"*

**Query-Key-Value:**
- **Query (Q):** "Tôi đang tìm kiếm gì?" — từ hiện tại đặt câu hỏi
- **Key (K):** "Tôi có thể cung cấp gì?" — mỗi từ trong chuỗi quảng cáo mình
- **Value (V):** "Thông tin thực sự của tôi" — nội dung thực sự khi được chọn

**Ví dụ:**
```
Câu: "The animal didn't cross the street because it was too tired"
→ "it" refers to "animal" hay "street"?
→ Query của "it" → Keys của "animal" và "street"
→ "animal" có score cao hơn → lấy Value của "animal"
```

#### 4.2.2 Scaled Dot-Product Attention — Derive

**Bước 1: Tạo Q, K, V từ input:**
```
Q = X × Wq    (projection matrix được học)
K = X × Wk
V = X × Wv
```

**Bước 2: Tính attention score:**
```
score(Q, K) = Q × Kᵀ     (dot product = mức độ tương đồng)
```

**Bước 3: Scale — Tại sao chia √d_k?**
```
score = Q × Kᵀ / √d_k
```
- Khi d_k lớn: dot product có variance lớn → softmax saturate → gradient vanish
- Chia √d_k để normalize variance về 1

**Bước 4: Softmax để ra trọng số:**
```
weights = softmax(score)    (tổng = 1, mỗi phần tử ∈ [0,1])
```

**Bước 5: Lấy weighted sum của Values:**
```
output = weights × V
```

**Công thức hoàn chỉnh:**
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

#### 4.2.3 Multi-Head Attention

**Ý tưởng:** Nhiều "đầu" attention học các loại relationship khác nhau song song
- Head 1: học syntactic relationship (chủ ngữ-động từ)
- Head 2: học coreference (đại từ → danh từ)
- Head 3: học positional relationship (từ kế bên)

```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ) × Wₒ
headᵢ = Attention(Q×Wqᵢ, K×Wkᵢ, V×Wvᵢ)
```

---

### 4.3 — Transformer Architecture

**Kiến trúc tổng thể (Vaswani et al., "Attention Is All You Need", 2017):**
```
Encoder:
  Input Embedding + Positional Encoding
  → N × [Multi-Head Self-Attention → Add & Norm → Feed Forward → Add & Norm]

Decoder:
  Output Embedding + Positional Encoding
  → N × [Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FF → Add & Norm]
```

#### 4.3.1 Positional Encoding — Tại Sao Cần?

**Vấn đề:** Attention không biết thứ tự từ!
- "Dog bites man" và "Man bites dog" có cùng attention nếu không có position info

**Sine/Cosine Encoding:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Tại sao sine/cosine?**
- Mỗi vị trí có pattern duy nhất
- Relative position có thể compute được từ tuyến tính combination
- Có thể generalize sang chuỗi dài hơn training data

#### 4.3.2 Add & Norm (Residual + Layer Norm)

```
output = LayerNorm(x + Sublayer(x))
```
- `x + Sublayer(x)`: residual connection (từ ResNet) → gradient flow
- LayerNorm: normalize theo features (khác BatchNorm normalize theo batch)

#### 4.3.3 Feed-Forward Network trong Transformer

```
FFN(x) = ReLU(x × W₁ + b₁) × W₂ + b₂
```
- Áp dụng **độc lập** cho mỗi position
- Mở rộng dimension rồi nén lại (thường 4× → 1×)
- Vai trò: xử lý thông tin sau khi attention đã "thu thập"

---

### 4.4 — BERT vs GPT — Hai Triết Lý Khác Nhau

**Cùng dùng Transformer nhưng khác hoàn toàn về cách train:**

| | BERT | GPT |
|--|------|-----|
| Kiến trúc | Encoder only | Decoder only |
| Direction | Bidirectional (nhìn cả 2 chiều) | Autoregressive (chỉ nhìn trái) |
| Pretraining | Masked Language Model (MLM) | Next token prediction |
| Ứng dụng | Classification, NER, QA | Text generation |
| Ví dụ | "Hôm nay trời [MASK] đẹp" → "rất" | "Hôm nay trời" → "rất" → "đẹp" → ... |

#### 4.4.1 BERT — Masked Language Model

**Pretraining task:**
```
Input:  "Tôi [MASK] cà phê mỗi [MASK]"
Target: "uống", "sáng"
```

**Tại sao bidirectional quan trọng?**
- "bank" trong "river bank" vs "bank account" cần context cả 2 phía
- GPT chỉ nhìn bên trái → không thể phân biệt

**Fine-tuning BERT:**
```
Pre-trained BERT + Classification Head → train với labeled data nhỏ
```

#### 4.4.2 GPT — Autoregressive Generation

**Pretraining task:** Dự đoán từ tiếp theo
```
"Tôi" → dự đoán "uống"
"Tôi uống" → dự đoán "cà"
"Tôi uống cà" → dự đoán "phê"
```

**Masked Self-Attention (Causal Mask):**
- Từ ở position t chỉ attend đến position ≤ t
- Implement bằng triangular mask:
```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

**Scaling law:** GPT-2 → GPT-3 → GPT-4: nhiều tham số hơn + nhiều data hơn → tốt hơn đột ngột

---

### 4.5 — Fine-tuning & Prompting

**Spectrum từ ít sang nhiều tùy chỉnh:**
```
Zero-shot → Few-shot → Prompt engineering → Fine-tuning → RLHF
```

**Zero-shot:** Chỉ mô tả task, không ví dụ
**Few-shot:** Cho 3-5 ví dụ trong prompt
**Fine-tuning:** Cập nhật toàn bộ weights với labeled data
**LoRA (Low-Rank Adaptation):** Chỉ train 1 phần nhỏ weights → hiệu quả hơn fine-tuning đầy đủ
**RLHF (Reinforcement Learning from Human Feedback):** Cách ChatGPT được train để hữu ích và an toàn

---

### 4.6 — Embeddings & Semantic Search

**Word embeddings — Tại sao cần?**
- One-hot encoding: "mèo" = [0,0,1,0,...] → không capture similarity
- Word2Vec: "mèo" ≈ "chó" (gần nhau trong không gian vector) → semantic meaning

**Từ Word2Vec đến LLM embeddings:**
- Word2Vec: mỗi từ có 1 vector cố định
- BERT/GPT: mỗi từ có vector **phụ thuộc context** (contextual embeddings)
- "bank" trong 2 câu khác nhau → 2 vector khác nhau

---

### 🏆 Milestone Project — Phase 4
**Xây Mini-GPT từ scratch (theo Karpathy)**
- Dataset: văn bản tiếng Việt (ví dụ: truyện Kiều)
- Implement: Transformer decoder, Attention, Positional Encoding
- Train character-level language model
- Generate text từ prompt
- **Mục tiêu:** Hiểu từng dòng code, generate được văn bản có vẻ hợp lệ

---

## ⏳ Phase 5 — MLOps & Production

> **Triết lý:** Model tốt mà không deploy được = không có giá trị.
> MLOps là cầu nối giữa Data Science và Software Engineering.

---

### Mục tiêu cuối phase
Deploy được một ML model lên production, monitor được performance,
và biết khi nào cần retrain.

---

### 5.1 — ML Lifecycle — Bức Tranh Toàn Cảnh

```
[Problem Definition]
       ↓
[Data Collection & Labeling]
       ↓
[EDA & Feature Engineering]
       ↓
[Model Development & Training]
       ↓
[Evaluation & Validation]
       ↓
[Deployment & Serving]
       ↓
[Monitoring & Maintenance]
       ↓ (khi model degraded)
[Retrain → back to top]
```

**Câu hỏi quan trọng:** Tại sao cần MLOps riêng, không phải DevOps thông thường?
- Code thay đổi → test là đủ
- Model thay đổi theo DATA — data thay đổi mà model không retrain → model xấu dần (model drift)

---

### 5.2 — Data Engineering cho ML

#### 5.2.1 Data Pipeline

**Vấn đề:** Làm sao đảm bảo data training và data production GIỐNG NHAU?

**Training-Serving Skew (lỗi rất phổ biến):**
```
Training:   feature = log(price + 1)
Production: feature = log(price)       ← quên +1 → bug ngầm
```

**Feature Store:**
- Nơi lưu trữ và chia sẻ features đã được compute
- Tránh compute feature trùng lặp ở nhiều nơi
- Đảm bảo training và serving dùng cùng logic
- Công cụ: Feast, Tecton, Vertex AI Feature Store

**Data Versioning:**
- Code có Git → Data cần DVC (Data Version Control)
- Biết model v3 được train trên data version nào

#### 5.2.2 Data Validation

**Tại sao cần validate data tự động?**
- Data từ upstream thay đổi format → model nhận input sai → crash hoặc predictions sai ngầm
- Cần detect sớm trước khi đến model

**Các kiểm tra cần có:**
```python
# Schema validation
assert "age" in dataframe.columns
assert dataframe["age"].dtype == int

# Range validation
assert dataframe["age"].between(0, 120).all()

# Distribution validation (so sánh với training data)
assert abs(dataframe["age"].mean() - training_mean) < threshold
```

**Công cụ:** Great Expectations, TFX Data Validation

---

### 5.3 — Experiment Tracking

**Vấn đề:** Chạy 50 experiments khác nhau — làm sao nhớ config nào cho kết quả tốt nhất?

**Những gì cần track mỗi experiment:**
- Hyperparameters: learning_rate, batch_size, n_layers,...
- Metrics: accuracy, loss, precision, recall trên train/val/test
- Artifacts: model weights, plots, confusion matrix
- Code version: Git commit hash
- Data version: DVC version

**MLflow — công cụ phổ biến nhất:**
```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 64)
    
    # ... train model ...
    
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

**Model Registry:** Lưu và version các model đã được validate
```
model_name/
  version_1/  → staging
  version_2/  → production
  version_3/  → development
```

---

### 5.4 — Model Deployment & Serving

#### 5.4.1 Các Hình Thức Serving

**Batch Serving:**
- Chạy predictions theo lịch (hàng giờ / hàng ngày)
- Ví dụ: recommend bài viết cho user mỗi đêm, email marketing hàng tuần
- Latency không quan trọng, throughput quan trọng

**Online Serving (Real-time):**
- REST API: nhận request → trả prediction trong milliseconds
- Ví dụ: fraud detection khi thanh toán, chatbot
- Latency cực quan trọng (< 100ms)

**Edge Serving:**
- Model chạy trên thiết bị (điện thoại, IoT)
- Không cần internet, private
- Model phải nhỏ (quantization, pruning)

#### 5.4.2 Containerization với Docker

**Vấn đề:** *"It works on my machine"*

**Dockerfile cho ML model:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model/ ./model/
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**FastAPI serving example:**
```python
from fastapi import FastAPI
import pickle

app = FastAPI()
model = pickle.load(open("model/model.pkl", "rb"))

@app.post("/predict")
def predict(data: InputData):
    features = preprocess(data)
    prediction = model.predict([features])
    return {"prediction": prediction[0]}
```

#### 5.4.3 Scaling & Load Balancing

**Khi nào cần scale?**
- 1000 requests/giây → 1 instance không đủ
- Kubernetes: tự động tăng/giảm số instances theo load

**A/B Testing:**
```
50% users → Model v1 (current production)
50% users → Model v2 (candidate)
→ Đo business metrics thực sau 1-2 tuần
→ Nếu v2 tốt hơn → promote lên production
```

---

### 5.5 — Monitoring & Alerting

#### 5.5.1 Các Loại Drift Cần Monitor

**Data Drift (Input Drift):**
- Distribution của input data thay đổi so với lúc train
- Ví dụ: app dùng ở nhóm tuổi mới, thói quen người dùng thay đổi
- Detect: statistical tests (KS test, Population Stability Index)

**Concept Drift:**
- Mối quan hệ giữa features và target thay đổi
- Ví dụ: pattern fraud mới xuất hiện, COVID làm thay đổi hành vi mua sắm
- Khó detect hơn — cần monitor model performance metrics

**Model Performance Drift:**
- Accuracy, precision, recall của model giảm dần
- Cần ground truth labels (thường có delay)

**Infrastructure Metrics:**
- Latency (P50, P95, P99)
- Throughput (requests/sec)
- Error rate (4xx, 5xx)
- CPU/Memory usage

#### 5.5.2 Monitoring Stack

```
Model → Predictions Log → [Evidently/Prometheus] → [Grafana Dashboard] → [PagerDuty Alert]
```

**Evidently AI:**
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=production_df)
report.save_html("drift_report.html")
```

---

### 5.6 — CI/CD cho Machine Learning

**CI/CD truyền thống (DevOps):**
```
Code push → Test → Build → Deploy
```

**CI/CD cho ML (MLOps):**
```
Code/Data/Config change → Test → Train → Evaluate → Build → Deploy → Monitor
```

**GitHub Actions pipeline ví dụ:**
```yaml
on: [push]
jobs:
  ml_pipeline:
    steps:
      - name: Run unit tests
      - name: Validate data schema
      - name: Train model
      - name: Evaluate model
        if: accuracy > 0.85  # Gate: chỉ deploy nếu đủ tốt
      - name: Build Docker image
      - name: Deploy to staging
      - name: Integration tests
      - name: Deploy to production
```

---

### 5.7 — Reproducibility & Governance

**Reproducibility là gì?**
- Chạy lại cùng experiment → ra cùng kết quả
- Cần: fixed random seed, versioned data, versioned code, same environment

**Model Card:**
- Document mô tả model: intended use, limitations, fairness evaluation
- Bắt buộc ở nhiều công ty và tổ chức

**Fairness & Bias:**
- Model có thể discriminate theo gender, race, age
- Cần đo và report performance trên các subgroups
- Công cụ: Fairlearn, AI Fairness 360

**Data Privacy:**
- GDPR: quyền được xóa dữ liệu → cần "machine unlearning"
- Không log sensitive data trong production

---

### 🏆 Milestone Project — Phase 5
**Deploy Heart Disease Predictor (từ Phase 1) lên production**
- Wrap model trong FastAPI
- Containerize với Docker
- Setup experiment tracking với MLflow
- Implement data drift monitoring với Evidently
- CI/CD pipeline với GitHub Actions
- **Bonus:** Deploy lên cloud (Heroku/Railway free tier)
- **Mục tiêu:** Một endpoint thực sự chạy được, monitor được

---

## ⏳ Track CV — Computer Vision Specialization (Tùy chọn)

> **Trạng thái:** Tùy chọn — nên làm sau Phase 4
> **Ghi chú:** Track này phù hợp nếu muốn chuyên sâu CV. Có thể học song song với Phase 5 hoặc sau.
> **Tài nguyên:** PyImageSearch, Szeliski "Computer Vision: Algorithms & Applications"

---

### Mục tiêu cuối track
Hiểu và implement được pipeline Computer Vision hoàn chỉnh: từ classical CV đến deep CV hiện đại.

---

### CV.1 — Classical CV (Nền Tảng)

- [ ] **OpenCV cơ bản** — xử lý ảnh, color spaces, filters
- [ ] **Edge detection** — Canny, Sobel (derive từ gradient)
- [ ] **Feature extraction** — SIFT, HOG (tại sao hoạt động?)

---

### CV.2 — Deep CV

- [ ] **CNN nâng cao** — U-Net cho segmentation, tại sao skip connections?
- [ ] **Object Detection** — YOLO (derive loss, IoU metric, Focal Loss)
- [ ] **Vision Transformer (ViT)** — patch embeddings, tại sao attention thay convolution?
- [ ] **Tracking** — SORT, DeepSORT

---

### 🏆 Milestone Project — Track CV
**License Plate Recognition pipeline (YOLO + OCR)** — hoặc dataset tương tự

---

## ⏳ Track VLM — Multimodal AI (Tùy chọn, Nâng Cao)

> **Trạng thái:** Sau Phase 4 + Track CV
> **Tài nguyên:** Karpathy PaliGemma video, Hugging Face VLM blog

---

### Mục tiêu cuối track
Hiểu cách kết hợp vision encoder và language model, tự xây được VLM đơn giản.

---

### VLM.1 — Checklist

- [ ] **Contrastive Learning** — CLIP: tại sao maximize cosine similarity của matching pairs?
- [ ] **SigLIP** — cải tiến gì so với CLIP? Sigmoid loss vs softmax
- [ ] **Bridging mechanism** — BLIP-2 Q-Former, tại sao cần information bottleneck?
- [ ] **Direct projection** — LLaVA style: MLP projection từ vision → language space
- [ ] **Fully integrated VLM** — PaliGemma (SigLIP + Gemma), causal vs non-causal attention mask
- [ ] **Visual Question Answering** — evaluate bằng exact-match, CLIP score

---

### 🏆 Milestone Project — Track VLM
**Xây VLM đơn giản:** nhận ảnh → trả lời câu hỏi về ảnh

---

## 📚 Tài Nguyên Theo Phase

| Phase | Tài nguyên chính |
|-------|-----------------|
| Phase 0–1 | StatQuest, 3Blue1Brown, sklearn docs |
| Phase 2 | 3Blue1Brown Neural Networks series, Karpathy Micrograd |
| Phase 3 | fast.ai, deeplearning.ai Deep Learning Specialization |
| Phase 4 | Karpathy "Zero to Hero", Sebastian Raschka "Build a LLM from Scratch" |
| Phase 5 | Chip Huyen "Designing ML Systems", "AI Engineering", MLflow docs |
| Track CV | PyImageSearch, Szeliski "Computer Vision: Algorithms & Applications" |
| Track VLM | Karpathy PaliGemma video, Hugging Face VLM blog |

---

## 📌 Gaps Cần Vá Trước Phase 2

### Gap 1: PCA từ bản chất *(ưu tiên cao nhất)*
Thứ tự học:
1. Covariance matrix là gì? Tại sao cần?
2. Eigenvector / Eigenvalue — ý nghĩa hình học
3. Tại sao eigenvector của covariance matrix = principal component?
4. SVD và mối quan hệ với PCA
5. Code từ scratch với NumPy
6. Verify với sklearn.PCA

### Gap 2: K-Means bản chất
1. Inertia = within-cluster sum of squares (derive)
2. Tại sao K-Means converge? (EM algorithm nhìn từ xa)
3. Limitations: số cluster, hình dạng cluster, outliers

### Gap 3: Gradient derivation thread còn bỏ ngỏ
- `Loss(w) = (6 - 2w)²` — derive gradient, có thể resume bất cứ lúc nào

---

## 📊 Tổng kết tiến độ

```
Phase 0   [██████████] 100% ✅ Toán nền tảng
Phase 1   [█████████░]  90% ✅ Classical ML (K-Means & PCA cần bổ sung bản chất)
Phase 2   [░░░░░░░░░░]   0% 🔄 Neural Networks — BẮT ĐẦU TIẾP THEO
Phase 3   [░░░░░░░░░░]   0% ⏳ Deep Learning Architectures
Phase 4   [░░░░░░░░░░]   0% ⏳ LLMs & Modern AI
Phase 5   [░░░░░░░░░░]   0% ⏳ MLOps & Production
Track CV  [░░░░░░░░░░]   0% ⏳ Computer Vision — Tùy chọn sau Phase 4
Track VLM [░░░░░░░░░░]   0% ⏳ Multimodal AI — Tùy chọn sau Track CV
```

---

## 🛠️ Nguyên tắc học (không thay đổi)

### 3 Lớp bắt buộc mỗi topic:
```
Lớp 1 — WHY   : Trực giác, ví dụ thực tế, tại sao cần
Lớp 2 — HOW   : Toán học, derive từ first principles
Lớp 3 — DO    : Code từ scratch → verify với sklearn/PyTorch
```

### Code style (bắt buộc — không dùng list comprehension):
```python
# ĐÚNG
distances = []
for centroid in centroids:
    distance = euclidean_distance(point, centroid)
    distances.append(distance)

# SAI
distances = [euclidean_distance(point, c) for c in centroids]
```

### Ký hiệu: `y = ax + b` | Dùng `a, b` thay cho `β₁, β₀`

---

*File này tổng hợp từ memory của Claude và skill file ml-tutor-eric.*
*Cập nhật lần cuối: Tháng 5/2026*