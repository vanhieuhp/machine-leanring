# Multi-Head Attention — Notes

## 1. Multi-head attention thực chất là gì

Chạy **nhiều attention song song**, mỗi cái nhìn input từ một góc độ khác nhau, rồi gộp kết quả lại.

Giống như khi đọc một câu — bạn vừa chú ý đến **ngữ pháp**, vừa chú ý đến **ngữ nghĩa**, vừa chú ý đến **vị trí** — cùng lúc, không phải lần lượt.

---

## 2. Vì sao split heads

Nếu dùng 1 attention duy nhất với `d_model=4`, model chỉ học được **1 kiểu quan hệ** giữa các token.

Split thành 2 heads (`head_dim=2`) → mỗi head có bộ Q, K, V riêng → **học 2 kiểu quan hệ độc lập** trên cùng input.

```
head 0: có thể học quan hệ cú pháp  (subject → verb)
head 1: có thể học quan hệ ngữ nghĩa (pronoun → noun)
```

Tổng số tham số không đổi — chỉ chia nhỏ không gian để nhìn đa chiều hơn.

---

## 3. Vì sao concat lại

Sau khi mỗi head tính xong, ta có:

```
head 0 output: [1, 3, 2]
head 1 output: [1, 3, 2]
```

Concat lại → `[1, 3, 4]` — khôi phục về `d_model` để tiếp tục qua các layer sau. Rồi nhân thêm `Wo` để **trộn thông tin từ các heads lại** thành một biểu diễn thống nhất.

---

## 4. Shape thay đổi thế nào

```
X                [1, 3, 4]     input
    ↓ Wq/Wk/Wv
Q, K, V          [1, 3, 4]     projected
    ↓ split_heads
Q_h, K_h, V_h   [1, 2, 3, 2]  (batch, heads, seq, head_dim)
    ↓ attention
output           [1, 2, 3, 2]
    ↓ transpose + reshape (concat)
output           [1, 3, 4]     về d_model
    ↓ Wo
output           [1, 3, 4]     final
```

---

## 5. Tại sao attention là matrix multiplication

Attention score giữa token `i` và token `j`:

```
score(i,j) = Q[i] · K[j]
```

Dot product đo **độ tương đồng hướng** giữa 2 vector — nếu Q[i] và K[j] cùng hướng thì score cao → token `i` chú ý nhiều đến token `j`.

Tính tất cả cặp cùng lúc bằng matrix multiplication:

```python
scores = Q @ K.T   # tính n² dot products trong 1 phép nhân
```

Nhanh hơn loop, GPU tối ưu cho phép toán này.

---

## 6. Một head học được gì

Mỗi head học một **bộ Wq, Wk, Wv riêng** → chiếu input vào không gian riêng → tập trung vào đặc trưng khác nhau.

Thực nghiệm trên BERT và GPT cho thấy các heads học được:

- Head chú ý đến **token liền kề** (local context)
- Head chú ý đến **token đầu câu** (positional)
- Head chú ý đến **từ cùng loại** (syntactic)
- Head chú ý đến **từ tham chiếu** (he/she → person)

Không ai lập trình điều này — model **tự học** trong quá trình training.
