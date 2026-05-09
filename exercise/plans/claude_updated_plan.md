# Lộ Trình Học Machine Learning — Eric Nguyen Van
> Cập nhật: Tháng 5/2026 | Dựa trên so sánh với AI Roadmap 2026

---

## Tổng quan lộ trình

| Phase | Tên | Trạng thái |
|-------|-----|------------|
| Phase 0 | Toán nền (Linear Algebra + Probability) | ✅ Hoàn thành |
| Phase 1 | Classical ML | ✅ Hoàn thành (một số topic cần bổ sung bản chất) |
| Phase 2 | Neural Networks từ scratch | 🔄 Đang bắt đầu |
| Phase 3 | Deep Learning Architectures | ⏳ Chưa bắt đầu |
| Phase 4 | LLMs & Modern AI | ⏳ Chưa bắt đầu |
| Phase 5 | MLOps & Production | ⏳ Chưa bắt đầu |
| Track CV | Computer Vision (Specialization) | ⏳ Tùy chọn — sau Phase 4 |
| Track VLM | Multimodal AI — Vision + Language | ⏳ Tùy chọn — sau Phase 4 + Track CV |

---

## Phase 0 — Toán nền ✅

**Trạng thái: Hoàn thành**

- ✅ Vectors, dot product, matrix multiplication
- ✅ Norms, projections
- ✅ Probability & Statistics cơ bản
- ✅ Đạo hàm, chain rule (dùng trong backprop sau này)

---

## Phase 1 — Classical ML ✅

**Trạng thái: Hoàn thành — milestone project: Heart Disease dataset (Logistic Regression, Accuracy 86.67%, Recall 82.61%)**

| Topic | Hoàn thành | Bản chất (3 lớp) |
|-------|-----------|-----------------|
| Linear Regression | ✅ | ✅ |
| Overfitting & Validation | ✅ | ✅ |
| Logistic Regression | ✅ | ✅ (MLE → BCE → gradient = P-y) |
| Decision Tree | ✅ | ✅ (Entropy, Information Gain) |
| Random Forest | ✅ | ✅ (Bias-variance, bootstrapping) |
| SVM | ✅ | ✅ (Margin maximize, Lagrangian) |
| K-Means | ✅ | ⚠️ Cần bổ sung: Inertia derive, convergence proof |
| PCA | ✅ | ⚠️ Cần bổ sung: Covariance matrix, eigenvector, SVD |

### 🔧 Việc cần làm trong Phase 1 (bổ sung bản chất):

**K-Means — cần học:**
- Inertia = within-cluster sum of squares (derive từ định nghĩa)
- Tại sao thuật toán hội tụ? (convergence proof)

**PCA — cần học từ đầu:**
- Covariance matrix là gì và tại sao dùng nó
- Eigenvector / eigenvalue — ý nghĩa hình học
- SVD và mối quan hệ với PCA
- Code PCA từ scratch với NumPy, sau đó verify với sklearn

---

## Phase 2 — Neural Networks từ scratch 🔄

**Trạng thái: Đang bắt đầu**
**Tài nguyên tham khảo:** 3Blue1Brown Neural Networks series, Andrej Karpathy Micrograd (dạy từ scratch, không assume đã biết)

### Checklist:

- [ ] **Perceptron & forward pass** — từ logistic regression → neuron đơn → nhiều neuron
- [ ] **Activation functions** — tại sao cần non-linearity? Sigmoid vs ReLU vs GELU
- [ ] **Computational graph** — biểu diễn phép tính dưới dạng đồ thị
- [ ] **Backpropagation** — chain rule → gradient flow qua từng node
- [ ] **Loss functions** — derive từ MLE/MAP (MSE, BCE, Cross-Entropy)
- [ ] **Optimization** — SGD → Momentum → Adam (derive từng cái)
- [ ] **Mini-batch gradient descent** — tại sao không dùng full-batch hay single-sample?
- [ ] **Weight initialization** — tại sao quan trọng? Xavier, He initialization

**Milestone project:** Tự build neural network từ scratch (chỉ NumPy) để classify MNIST hoặc tương tự

---

## Phase 3 — Deep Learning Architectures ⏳

**Trạng thái: Chưa bắt đầu**

### Checklist:

- [ ] **CNN** — Convolution operation, receptive field, parameter sharing
- [ ] **Pooling** — tại sao cần? Max vs Average pooling
- [ ] **Kiến trúc nổi tiếng** — VGG → ResNet (skip connections, tại sao giải quyết vanishing gradient)
- [ ] **RNN** — sequential data, hidden state
- [ ] **BPTT** — Backpropagation through time
- [ ] **Vanishing gradient problem** — tại sao xảy ra, LSTM giải quyết thế nào
- [ ] **Batch Normalization** — derive, tại sao stabilize training
- [ ] **Dropout** — tại sao giảm overfitting, mối quan hệ với ensemble

**Milestone project:** Image classifier với CNN từ scratch (PyTorch), so sánh ResNet với plain CNN

---

## Phase 4 — LLMs & Modern AI ⏳

**Trạng thái: Chưa bắt đầu**
**Tài nguyên:** Andrej Karpathy "Zero to Hero", Sebastian Raschka "Build a LLM from Scratch"

### Checklist:

- [ ] **Tokenization** — BPE, WordPiece, tại sao không dùng word-level?
- [ ] **Word Embeddings** — Word2Vec derive, tại sao embedding hoạt động?
- [ ] **Attention mechanism** — scaled dot-product, Q/K/V từ đâu ra?
- [ ] **Multi-Head Attention** — tại sao cần nhiều head?
- [ ] **Positional Encoding** — sinusoidal → RoPE, tại sao cần?
- [ ] **Transformer architecture** — encoder vs decoder vs encoder-decoder
- [ ] **BERT** — Masked Language Modeling, bidirectional context
- [ ] **GPT** — autoregressive, causal attention mask
- [ ] **Fine-tuning** — SFT, LoRA (parameter-efficient), tại sao freeze base model?
- [ ] **Alignment** — RLHF, DPO — tại sao cần align?
- [ ] **Quantization** — FP16 → INT8/INT4, GGUF format

**Milestone project:** Code một GPT nhỏ từ scratch (Karpathy style), train trên text đơn giản

---

## Phase 5 — MLOps & Production ⏳

**Trạng thái: Chưa bắt đầu**
**Tài nguyên:** Chip Huyen "Designing Machine Learning Systems", "AI Engineering"

### Checklist:

- [ ] **Model serving** — FastAPI, REST endpoint cho ML model
- [ ] **Containerization** — Docker cơ bản cho ML workflows
- [ ] **Experiment tracking** — MLflow: log metrics, parameters, artifacts
- [ ] **Data versioning** — DVC hoặc tương tự
- [ ] **Model monitoring** — data drift detection, performance degradation
- [ ] **Inference optimization** — ONNX, quantization cho deployment
- [ ] **vLLM & LLM serving** — PagedAttention, continuous batching
- [ ] **AI Security cơ bản** — prompt injection, adversarial inputs

**Milestone project:** Deploy model từ Phase 4 lên API endpoint có monitoring

---

## Track CV — Computer Vision Specialization ⏳ (Tùy chọn)

**Trạng thái: Tùy chọn — nên làm sau Phase 4**
**Ghi chú:** Track này phù hợp nếu muốn chuyên sâu CV. Có thể học song song với Phase 5 hoặc sau.

### Checklist:

**Classical CV (nền tảng):**
- [ ] OpenCV cơ bản — xử lý ảnh, color spaces, filters
- [ ] Edge detection — Canny, Sobel (derive từ gradient)
- [ ] Feature extraction — SIFT, HOG (tại sao hoạt động?)

**Deep CV:**
- [ ] CNN nâng cao — U-Net cho segmentation, tại sao skip connections?
- [ ] Object Detection — YOLO (derive loss, IoU metric, Focal Loss)
- [ ] Vision Transformer (ViT) — patch embeddings, tại sao attention thay convolution?
- [ ] Tracking — SORT, DeepSORT

**Milestone project:** License Plate Recognition pipeline (YOLO + OCR) — hoặc dataset tương tự

---

## Track VLM — Multimodal AI ⏳ (Tùy chọn, nâng cao)

**Trạng thái: Sau Phase 4 + Track CV**

### Checklist:

- [ ] **Contrastive Learning** — CLIP: tại sao maximize cosine similarity của matching pairs?
- [ ] **SigLIP** — cải tiến gì so với CLIP? Sigmoid loss vs softmax
- [ ] **Bridging mechanism** — BLIP-2 Q-Former, tại sao cần information bottleneck?
- [ ] **Direct projection** — LLaVA style: MLP projection từ vision → language space
- [ ] **Fully integrated VLM** — PaliGemma (SigLIP + Gemma), causal vs non-causal attention mask
- [ ] **Visual Question Answering** — evaluate bằng exact-match, CLIP score

**Milestone project:** Xây VLM đơn giản: nhận ảnh → trả lời câu hỏi về ảnh

---

## Tài nguyên theo Phase

| Phase | Tài nguyên chính |
|-------|-----------------|
| Phase 0-1 | StatQuest, 3Blue1Brown, sklearn docs |
| Phase 2 | 3Blue1Brown Neural Networks, Karpathy Micrograd |
| Phase 3 | fast.ai, deeplearning.ai Deep Learning Specialization |
| Phase 4 | Karpathy "Zero to Hero", Sebastian Raschka "Build a LLM" |
| Phase 5 | Chip Huyen "Designing ML Systems", MLflow docs |
| Track CV | PyImageSearch, Szeliski "Computer Vision: Algorithms & Applications" |
| Track VLM | Karpathy PaliGemma video, Hugging Face VLM blog |

---

## Nguyên tắc học xuyên suốt

1. **Luôn derive trước** — không chấp nhận công thức "từ trên trời rơi xuống"
2. **Code từ scratch trước** — sklearn/PyTorch chỉ để verify
3. **Hiểu một topic thật sâu** trước khi sang topic tiếp
4. **Kết nối ngược** — mỗi topic mới phải liên kết được với kiến thức cũ