# Personal Note: Lý thuyết Hồi quy trong Học máy có Giám sát 


## 1. Giới thiệu về Hồi quy (Regression)
- **Học máy có giám sát (Supervised Learning)**: Đây là một loại học máy nơi mô hình được huấn luyện trên dữ liệu đã được gắn nhãn, nghĩa là mỗi mẫu dữ liệu đầu vào (features) đi kèm với giá trị đầu ra đúng (labels hoặc targets). Mục tiêu là học cách dự đoán đầu ra cho dữ liệu mới dựa trên các mẫu đã học. Quy trình bao gồm: thu thập dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá, và triển khai. Supervised Learning khác với unsupervised learning (không có nhãn) ở chỗ nó tập trung vào dự đoán chính xác dựa trên dữ liệu đã biết.
  
- **Hồi quy (Regression)**: Là một phần của supervised learning, dùng để dự đoán giá trị liên tục (continuous) thay vì giá trị rời rạc. Regression giả định rằng có mối quan hệ toán học giữa các features và target, và mô hình cố gắng ước lượng mối quan hệ này. Các giả định cơ bản bao gồm: tuyến tính (nếu là linear), độc lập giữa các quan sát, và phân bố chuẩn của lỗi. Nếu vi phạm các giả định này, cần sử dụng các kỹ thuật biến đổi dữ liệu hoặc mô hình khác.

- **So sánh với phân loại (Classification)**: Regression dự đoán số thực (ví dụ: 3.14, 100.5), trong khi classification dự đoán nhãn hạng mục (ví dụ: "có/không", "mèo/chó"). Regression thường sử dụng hàm mất mát như MSE để đo lường sai số liên tục, còn classification dùng cross-entropy cho xác suất.

**Ví dụ thực tế**: Trong dự đoán giá nhà, regression có thể dự đoán giá chính xác là 250.000 USD dựa trên diện tích, số phòng, và vị trí. Ngược lại, classification có thể phân loại nhà là "rẻ" hoặc "đắt" dựa trên ngưỡng giá.

![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424231192244523108/9BAIBbkhARERERGmJ81kiotRi0ZaIiIiIiIiIiIhIQ7gRGREREREREREREZGGsGhLREREREREREREpCEs2hIRERERERERERFpCIu2RERERERERERERBrCoi0RERERERERERGRhrBoS0RERERERERERKQhLNoSERERERERERERaQiLtkREREREREREREQa8v8B8sxTSsfKnPUAAAAASUVORK5CYII.png?ex=68e33229&is=68e1e0a9&hm=4778f3b7167fda714de35d7fe5291967de07c23fa5f5fb2a9cfb2c624a8f5654&=&format=webp&quality=lossless&width=1730&height=738)

---




## 2. Linear Regression
- **Khái niệm**: Linear Regression giả định mối quan hệ tuyến tính giữa các features và target. Nó tìm đường thẳng (trong không gian 2D) hoặc siêu phẳng (trong không gian đa chiều) sao cho tổng sai số giữa dự đoán và thực tế là nhỏ nhất. Có hai loại: Simple Linear Regression (một feature) và Multiple Linear Regression (nhiều features). Ưu điểm: Dễ hiểu, tính toán nhanh. Nhược điểm: Không phù hợp với dữ liệu phi tuyến tính.

- **Công thức**:
  \[
  y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n + \epsilon
  \]
  - \( y \): Target (giá trị dự đoán).
  - \( w_0 \): Intercept (giá trị y khi tất cả x=0).
  - \( w_i \): Coefficients (độ dốc, cho biết mức độ ảnh hưởng của feature \( x_i \)).
  - \( \epsilon \): Lỗi ngẫu nhiên (residuals), giả định phân bố chuẩn với kỳ vọng 0.

- **Hàm mất mát (Loss Function)**: Mean Squared Error (MSE) là phổ biến nhất vì phạt lỗi lớn mạnh hơn:
  \[
  MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
  \]
  - Để tối ưu, sử dụng Ordinary Least Squares (OLS) cho giải đóng (closed-form) hoặc Gradient Descent cho giải lặp. Gradient Descent cập nhật weights theo công thức:
    \[
    w_j := w_j - \alpha \frac{\partial}{\partial w_j} MSE
    \]
    - \( \alpha \): Learning rate (bước học).

- **Tối ưu hóa**: Gradient Descent bắt đầu từ weights ngẫu nhiên, tính gradient của loss, và di chuyển ngược chiều gradient để giảm loss. Có các biến thể như Stochastic GD (dùng một mẫu) hoặc Mini-batch GD (dùng batch nhỏ) để tăng tốc.

**Ví dụ thực tế**: Dự đoán lương dựa trên kinh nghiệm. Giả sử dữ liệu: Kinh nghiệm (x) = [1, 2, 3, 4] năm, Lương (y) = [30k, 40k, 50k, 60k] USD. Mô hình có thể tìm \( y = 20k + 10k \times x \). Với x=5, dự đoán y=70k. Nếu thêm feature như bằng cấp, thành Multiple Linear Regression.
**Code minh họa**
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dữ liệu ví dụ
X = np.array([[1], [2], [3], [4]])  # Kinh nghiệm (năm)
y = np.array([30, 40, 50, 60])      # Lương (nghìn USD)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X, y)

# Dự đoán
X_test = np.array([[5]])
y_pred = model.predict(X_test)
print(f"Dự đoán lương cho 5 năm kinh nghiệm: {y_pred[0]:.1f}k USD")

# Vẽ biểu đồ
plt.scatter(X, y, color='blue', label='Dữ liệu')
plt.plot(X, model.predict(X), color='red', label='Đường hồi quy')
plt.xlabel('Kinh nghiệm (năm)')
plt.ylabel('Lương (nghìn USD)')
plt.legend()
plt.show()
```
![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424231842940321855/wNfz7Ux9kwwTwAAAABJRU5ErkJggg.png?ex=68e332c4&is=68e1e144&hm=14af6aaff80c1ccfcd9d1915c151afe9de2eb6502691a557ec636a284e93b52a&=&format=webp&quality=lossless&width=704&height=541)
---




## 3. Polynomial Regression
- **Khái niệm**: Khi dữ liệu không tuyến tính, Polynomial Regression sử dụng các lũy thừa của features để tạo đường cong. Đây thực chất là linear regression trên không gian features đã biến đổi (feature engineering). Độ phức tạp tăng theo bậc polynomial (degree), nhưng dễ dẫn đến overfitting nếu degree cao hoặc underfitting nếu thấp.

- **Công thức**:
  \[
  y = w_0 + w_1x + w_2x^2 + \dots + w_nx^n + \epsilon
  \]
  - Để huấn luyện, biến đổi x thành [x, x², x³,...] rồi áp dụng linear regression.

- **Vấn đề**: Overfitting xảy ra khi mô hình học noise thay vì pattern thực, dẫn đến hiệu suất kém trên dữ liệu mới. Underfitting là khi mô hình quá đơn giản, bỏ lỡ pattern. Để chọn degree tối ưu, dùng cross-validation.

**Ví dụ thực tế**: Dự đoán tốc độ xe dựa trên thời gian. Dữ liệu: Thời gian (x) = [1,2,3,4,5], Tốc độ (y) = [10, 20, 25, 20, 10] km/h (hình parabol). Mô hình polynomial degree 2: \( y = 5 + 15x - 2x^2 \). Với x=6, dự đoán y=3 km/h (giảm tốc).
**Code minh họa**:
```
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Dữ liệu ví dụ
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 25, 20, 10])

# Tạo mô hình Polynomial Regression (degree=2)
polyreg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
polyreg.fit(X, y)

# Dự đoán
X_test = np.array([[6]])
y_pred = polyreg.predict(X_test)
print(f"Dự đoán tốc độ tại t=6: {y_pred[0]:.1f} km/h")

# Vẽ biểu đồ
X_plot = np.linspace(1, 6, 100).reshape(-1, 1)
y_plot = polyreg.predict(X_plot)
plt.scatter(X, y, color='blue', label='Dữ liệu')
plt.plot(X_plot, y_plot, color='red', label='Đường polynomial')
plt.xlabel('Thời gian (s)')
plt.ylabel('Tốc độ (km/h)')
plt.legend()
plt.show()
```
**Hình ảnh minh họa**: Biểu đồ đường cong polynomial regression.
![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424234272943898664/WMHAfS8LOMQAAAABJRU5ErkJggg.png?ex=68e33507&is=68e1e387&hm=995da0e2cd28284cc35b32bc575cb2bd437a65966a2b94c587c0813ef99126c8&=&format=webp&quality=lossless&width=708&height=541)
---




## 4. Regularization
- **Mục đích**: Regularization thêm phạt vào loss function để kiểm soát độ phức tạp, ngăn overfitting bằng cách khuyến khích weights nhỏ hoặc bằng 0. Nó cân bằng giữa fitting dữ liệu và generalization.

- **Các loại**:
  - **Lasso Regression (L1)**: Phạt tổng giá trị tuyệt đối của weights:
    \[
    Loss = MSE + \lambda \sum_{i=1}^n |w_i|
    \]
    - Lợi ích: Làm một số weights = 0, tự động chọn features (sparse model). Phù hợp khi nhiều features không quan trọng.
  - **Ridge Regression (L2)**: Phạt tổng bình phương weights:
    \[
    Loss = MSE + \lambda \sum_{i=1}^n w_i^2
    \]
    - Lợi ích: Làm weights nhỏ nhưng không bằng 0, xử lý multicollinearity (features tương quan cao).
  - **Elastic Net**: Kết hợp L1 và L2:
    \[
    Loss = MSE + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2
    \]
    - Phù hợp khi cần cả feature selection và xử lý tương quan.

- **Tham số \( \lambda \)**: Lambda (alpha) kiểm soát mức phạt. Lambda=0: Không regularization (linear regression). Lambda lớn: Underfitting. Chọn lambda qua cross-validation.

**Ví dụ thực tế**: Trong dự đoán giá nhà với 100 features (diện tích, phòng, tuổi nhà, màu sắc,...). Lasso có thể loại bỏ "màu sắc" (weight=0) vì không quan trọng, còn Ridge làm weight của features ít ảnh hưởng nhỏ lại.
**Code**
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# --- 1. Tạo dữ liệu giả lập ---
# Tạo bộ dữ liệu với 100 features.
# Chỉ 10 features đầu tiên là thực sự hữu ích (n_informative=10).
# Các features còn lại là nhiễu.
X, y, w = make_regression(
    n_samples=150,
    n_features=100,
    n_informative=10,
    noise=15,
    coef=True,
    random_state=42
)

# --- 2. Huấn luyện các mô hình ---

# a) Mô hình hồi quy tuyến tính (Không có Regularization)
lr = LinearRegression()
lr.fit(X, y)

# b) Mô hình Ridge (L2 Regularization)
# Alpha là tham số điều chỉnh độ mạnh của regularization. Alpha càng lớn, penalty càng mạnh.
ridge = Ridge(alpha=10)
ridge.fit(X, y)

# c) Mô hình Lasso (L1 Regularization)
# Alpha ở đây cũng điều chỉnh độ mạnh.
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)


# --- 3. Trực quan hóa và so sánh các hệ số (coefficients) ---

# Tạo biểu đồ
plt.figure(figsize=(14, 8))
plt.title('So sánh hệ số của các mô hình', fontsize=16)

# Vẽ các hệ số thực tế (ground truth) mà chúng ta đã tạo ra
plt.plot(w, alpha=0.7, linestyle='none', marker='o', markersize=7, color='red', label='Hệ số thực tế (Ground Truth)')

# Vẽ các hệ số của mô hình Linear Regression
plt.plot(lr.coef_, alpha=0.6, linestyle='none', marker='s', markersize=7, color='blue', label='Linear Regression (Không Regularization)')

# Vẽ các hệ số của mô hình Ridge
plt.plot(ridge.coef_, alpha=0.8, linestyle='none', marker='^', markersize=7, color='green', label='Ridge (L2 Regularization)')

# Vẽ các hệ số của mô hình Lasso
plt.plot(lasso.coef_, alpha=0.9, linestyle='none', marker='x', markersize=7, color='purple', label='Lasso (L1 Regularization)')


# Tùy chỉnh biểu đồ
plt.xlabel('Chỉ số của Feature', fontsize=12)
plt.ylabel('Giá trị của hệ số (Weight)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```
**Hình ảnh minh họa**: Bảng so sánh Ridge vs Lasso.

![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424237480856518748/rs455xxlNpuV2WxWGRkZauPGjaqoqGjE6xRCCCFEYGhKBegWlRBCCCGEEEIIIYSYMmRPISGEEEIIIYQQQogZSAaFhBBCCCGEEEIIIWYgGRQSQgghhBBCCCGEmIFkUEgIIYQQQgghhBBiBpJBISGEEEIIIYQQQogZSAaFhBBCCCGEEEIIIWYgGRQSQgghhBBCCCGEmIFkUEgIIYQQQgghhBBiBpJBISGEEEIIIYQQQogZSAaFhBBCCCGEEEIIIWYgGRQSQgghhBBCCCGEmIFkUEgIIYQQQgghhBBiBpJBISGEEEIIIYQQQogZ6P8HrAUosrtnbisAAAAASUVORK5CYII.png?ex=68e33804&is=68e1e684&hm=2a19aba7c6ebbb196bc2be7fd7b49f50bd36a2f201fb10113c36a757e6a4b654&=&format=webp&quality=lossless&width=1240&height=760)
---




## 5. Đánh giá mô hình
- **Các chỉ số đánh giá**: Đo lường độ chính xác và generalization. Không chỉ dùng trên train set mà phải dùng test set hoặc cross-validation để tránh bias.

  - **Mean Absolute Error (MAE)**: Trung bình sai số tuyệt đối, dễ diễn giải:
    \[
    MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
    \]
    - Phù hợp khi lỗi lớn không quá nghiêm trọng.

  - **Mean Squared Error (MSE)**: Phạt lỗi lớn mạnh hơn.

  - **Root Mean Squared Error (RMSE)**: Căn bậc hai MSE, có đơn vị giống target:
    \[
    RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
    \]

  - **R-squared (\( R^2 \))**: Tỷ lệ biến thiên target được giải thích bởi mô hình:
    \[
    R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
    \]
    - \( R^2 = 1 \): Hoàn hảo; \( R^2 = 0 \): Không tốt hơn mean; âm: Tệ hơn mean.

- **Cross-Validation**: K-Fold CV chia dữ liệu thành K folds, huấn luyện trên K-1 folds, test trên 1 fold, lặp K lần. Giúp ước lượng hiệu suất trung bình, tránh overfitting.

**Ví dụ thực tế**: Với mô hình dự đoán giá nhà, MSE=10000 nghĩa là trung bình sai số bình phương 10000 USD². RMSE=100 USD dễ hiểu hơn. Nếu \( R^2 = 0.85 \), mô hình giải thích 85% biến thiên giá nhà.
**Code**:
```
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Tạo dữ liệu giả lập ---
# Tạo mối quan hệ tuyến tính giữa diện tích và giá nhà, sau đó thêm nhiễu ngẫu nhiên
np.random.seed(42) # Để đảm bảo kết quả luôn giống nhau mỗi khi chạy
dien_tich = np.random.rand(100, 1) * 100 + 50  # Tạo 100 ngôi nhà có diện tích từ 50m² đến 150m²
# Giá nhà = 50 (giá cơ bản) + 3.5 * diện tích + nhiễu
gia_thuc_te = 50 + 3.5 * dien_tich + np.random.randn(100, 1) * 40

# --- 2. Huấn luyện mô hình hồi quy tuyến tính ---
model = LinearRegression()
model.fit(dien_tich, gia_thuc_te)

# --- 3. Lấy kết quả dự đoán từ mô hình ---
gia_du_doan = model.predict(dien_tich)

# --- 4. Tính toán các chỉ số đánh giá ---
# Mean Squared Error (MSE)
mse = mean_squared_error(gia_thuc_te, gia_du_doan)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared (R²)
r2 = r2_score(gia_thuc_te, gia_du_doan)



# ---  Trực quan hóa kết quả ---
plt.figure(figsize=(10, 6))
plt.scatter(dien_tich, gia_thuc_te, color='blue', label='Giá thực tế', alpha=0.6)
plt.plot(dien_tich, gia_du_doan, color='red', linewidth=2, label='Giá dự đoán bởi mô hình')
plt.title('Dự đoán giá nhà dựa trên Diện tích', fontsize=16)
plt.xlabel('Diện tích (m²)', fontsize=12)
plt.ylabel('Giá nhà (nghìn USD)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
```
**Hình ảnh minh họa**: Biểu đồ các metrics đánh giá.
![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424238155996856461/jB0rBcuvWbIgAAAABJRU5ErkJggg.png?ex=68e338a5&is=68e1e725&hm=6b123f5fa2a008ce262738b339be49a9fa77f1522558846e918fa7b3f243b33c&=&format=webp&quality=lossless&width=1069&height=694)
---

## 6. Tổng kết
- **Linear Regression**: Nền tảng cho dữ liệu tuyến tính, dễ triển khai nhưng giới hạn với phi tuyến.
- **Polynomial Regression**: Linh hoạt hơn cho dữ liệu cong, nhưng cần kiểm soát degree để tránh overfitting.
- **Regularization**: Công cụ mạnh mẽ để cân bằng bias-variance, đặc biệt với dữ liệu cao chiều.
- **Đánh giá**: Luôn dùng nhiều metrics và cross-validation để đảm bảo mô hình robust.
