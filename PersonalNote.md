

## **1. Giới thiệu về Hồi quy (Regression)**

### **1.1. Học máy có Giám sát (Supervised Learning)**
- **Định nghĩa**: Học máy có giám sát là một nhánh của học máy, trong đó mô hình được huấn luyện trên một tập dữ liệu đã được "gắn nhãn". Điều này có nghĩa là mỗi điểm dữ liệu đầu vào (features) đều đi kèm với một kết quả đầu ra (label hoặc target) chính xác.
- **Mục tiêu**: Mục tiêu cuối cùng là học ra một hàm ánh xạ (mapping function) có thể dự đoán giá trị đầu ra cho những dữ liệu mới chưa từng thấy trước đây.
- **Quy trình làm việc điển hình**:
    1.  **Thu thập dữ liệu**: Tập hợp dữ liệu có chứa cả đầu vào và đầu ra mong muốn.
    2.  **Tiền xử lý dữ liệu**: Làm sạch, xử lý các giá trị thiếu, và chuẩn hóa dữ liệu để mô hình hoạt động hiệu quả.
    3.  **Phân chia dữ liệu**: Chia dữ liệu thành tập huấn luyện (training set) và tập kiểm thử (testing set) để đánh giá hiệu suất của mô hình một cách khách quan.
    4.  **Huấn luyện mô hình**: Sử dụng tập huấn luyện để "dạy" cho thuật toán tìm ra các mẫu và mối quan hệ trong dữ liệu.
    5.  **Đánh giá mô hình**: Dùng tập kiểm thử để xem mô hình dự đoán tốt đến đâu trên dữ liệu mới.
    6.  **Tinh chỉnh và Triển khai**: Tối ưu hóa mô hình và đưa vào sử dụng thực tế.
- **So sánh với các loại học máy khác**:
    - **Học máy không giám sát (Unsupervised Learning)**: Làm việc với dữ liệu không có nhãn, mục tiêu là tự khám phá ra các cấu trúc hoặc mẫu ẩn, ví dụ như phân cụm khách hàng (clustering).
    - **Học tăng cường (Reinforcement Learning)**: Mô hình học bằng cách tương tác với một môi trường và nhận phần thưởng hoặc hình phạt, ví dụ như huấn luyện một AI để chơi game.

### **1.2. Hồi quy (Regression)**
- **Định nghĩa**: Hồi quy là một nhiệm vụ thuộc học máy có giám sát, tập trung vào việc dự đoán một giá trị đầu ra **liên tục** (continuous). Giá trị liên tục có thể là bất kỳ số thực nào trong một khoảng, ví dụ như giá nhà, nhiệt độ, hoặc doanh thu.
- **Bản chất**: Cốt lõi của hồi quy là giả định rằng có một mối quan hệ toán học nào đó giữa các biến đầu vào (features) và biến đầu ra (target). Mô hình hồi quy sẽ cố gắng "học" và ước lượng mối quan hệ này.
- **Các giả định cơ bản (đặc biệt quan trọng đối với Linear Regression)**:
    - **Tuyến tính (Linearity)**: Có mối quan hệ tuyến tính giữa các feature và target.
    - **Độc lập (Independence)**: Các quan sát (mẫu dữ liệu) là độc lập với nhau.
    - **Phương sai không đổi của sai số (Homoscedasticity)**: Phương sai của sai số (phần còn lại sau khi dự đoán) là như nhau cho mọi giá trị của biến độc lập.
    - **Phân phối chuẩn của sai số (Normality of Residuals)**: Sai số của mô hình tuân theo phân phối chuẩn.
    *Việc vi phạm các giả định này có thể làm giảm độ tin cậy của mô hình, và khi đó chúng ta cần các kỹ thuật nâng cao hơn để xử lý.*

### **1.3. So sánh Hồi quy và Phân loại (Regression vs. Classification)**

| Tiêu chí | Hồi quy (Regression) | Phân loại (Classification) |
| :--- | :--- | :--- |
| **Loại đầu ra** | Giá trị liên tục (số thực) | Nhãn rời rạc (hạng mục) |
| **Ví dụ** | Dự đoán giá nhà (250.000 USD), nhiệt độ (25.5°C) | Phân loại email ("spam"/"không spam"), nhận diện ảnh ("mèo"/"chó") |
| **Câu hỏi trả lời** | "Bao nhiêu?" (How much/How many?) | "Loại nào?" (Which class/What kind?) |
| **Hàm mất mát phổ biến**| Mean Squared Error (MSE), Mean Absolute Error (MAE) | Cross-Entropy Loss, Hinge Loss |
| **Thuật toán phổ biến** | Linear Regression, Ridge, Lasso | Logistic Regression, SVM, Decision Tree |

**Ví dụ thực tế**: Trong việc dự đoán giá của một ngôi nhà:
- **Hồi quy** có thể dự đoán giá chính xác là **251.350 USD** dựa trên diện tích, số phòng ngủ, và vị trí.
- **Phân loại** có thể phân loại ngôi nhà đó là **"giá rẻ"**, **"trung bình"**, hoặc **"đắt"** dựa trên một ngưỡng giá định trước.

![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424231192244523108/9BAIBbkhARERERGmJ81kiotRi0ZaIiIiIiIiIiIhIQ7gRGREREREREREREZGGsGhLREREREREREREpCEs2hIRERERERERERFpCIu2RERERERERERERBrCoi0RERERERERERGRhrBoS0RERERERERERKQhLNoSERERERERERERaQiLtkREREREREREREQa8v8B8sxTSsfKnPUAAAAASUVORK5CYII.png?ex=68e33229&is=68e1e0a9&hm=4778f3b7167fda714de35d7fe5291967de07c23fa5f5fb2a9cfb2c624a8f5654&=&format=webp&quality=lossless&width=1730&height=738)

---

## **2. Hồi quy Tuyến tính (Linear Regression)**

### **2.1. Khái niệm**
Linear Regression là một trong những thuật toán đơn giản và nền tảng nhất. Nó hoạt động dựa trên giả định rằng có một mối quan hệ tuyến tính giữa các biến đầu vào (features) và biến đầu ra (target).
- **Mục tiêu**: Tìm ra một đường thẳng (trong không gian 2D) hoặc một siêu phẳng (trong không gian đa chiều) "phù hợp nhất" (best fit) với dữ liệu. "Phù hợp nhất" có nghĩa là tổng khoảng cách bình phương từ các điểm dữ liệu thực tế đến đường thẳng/siêu phẳng đó là nhỏ nhất.
- **Phân loại**:
    - **Simple Linear Regression**: Chỉ có một biến đầu vào (ví dụ: dự đoán lương chỉ dựa vào số năm kinh nghiệm).
    - **Multiple Linear Regression**: Có nhiều hơn một biến đầu vào (ví dụ: dự đoán lương dựa vào kinh nghiệm, bằng cấp, và vị trí làm việc).
- **Ưu điểm**: Dễ diễn giải, tính toán nhanh, và là nền tảng tốt để hiểu các mô hình phức tạp hơn.
- **Nhược điểm**: Không hiệu quả với các mối quan hệ phi tuyến tính phức tạp.

### **2.2. Công thức toán học**
Công thức tổng quát của Hồi quy tuyến tính đa biến:
$$
y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n + \epsilon
$$
- $y$: Biến mục tiêu (target) - giá trị chúng ta muốn dự đoán.
- $x_1, x_2, \dots, x_n$: Các biến độc lập (features) - thông tin đầu vào.
- $w_0$: Hệ số chặn (Intercept hoặc Bias) - là giá trị của $y$ khi tất cả các feature $x_i$ bằng 0.
- $w_1, w_2, \dots, w_n$: Các hệ số hồi quy (Coefficients hoặc Weights) - đại diện cho độ dốc. Mỗi $w_i$ cho biết mức độ ảnh hưởng của feature $x_i$ lên $y$.
- $\epsilon$: Sai số ngẫu nhiên (Residuals) - phần biến thiên của $y$ không thể giải thích được bởi mô hình.

### **2.3. Hàm mất mát (Loss Function) và Tối ưu hóa**
- **Mean Squared Error (MSE)**: Là hàm mất mát phổ biến nhất cho hồi quy. Nó tính trung bình của tổng các bình phương sai số giữa giá trị thực tế ($y_i$) và giá trị dự đoán ($\hat{y}_i$).
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
- **Phương pháp tối ưu hóa Gradient Descent**: Thuật toán cập nhật các trọng số $w_j$ để giảm thiểu hàm mất mát.
$$
w_j := w_j - \alpha \frac{\partial}{\partial w_j} \text{MSE}
$$
- $\alpha$ (Learning rate - Tốc độ học): Là một siêu tham số quyết định "bước đi" lớn hay nhỏ trong mỗi lần cập nhật.

### **2.4. Ví dụ và Code minh họa**
**Ví dụ thực tế**: Dự đoán mức lương dựa trên số năm kinh nghiệm. Mô hình có thể tìm ra mối quan hệ là $y = 20k + 10k \times x$. Với một người có 5 năm kinh nghiệm, mô hình sẽ dự đoán lương là $20k + 10k \times 5 = 70k$ USD.

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

## **3. Hồi quy Đa thức (Polynomial Regression)**

### **3.1. Khái niệm**
Khi mối quan hệ giữa feature và target không phải là một đường thẳng mà là một đường cong, Polynomial Regression giải quyết vấn đề này bằng cách tạo ra các feature mới là lũy thừa của các feature ban đầu, sau đó áp dụng mô hình Linear Regression trên tập feature đã được mở rộng này.

### **3.2. Công thức**
Mô hình hồi quy đa thức bậc $d$ với một feature $x$:
$$
y = w_0 + w_1x + w_2x^2 + \dots + w_dx^d + \epsilon
$$
Để huấn luyện, chúng ta biến đổi feature $x$ thành một tập hợp các feature mới là $[x, x^2, \dots, x^d]$ và áp dụng Linear Regression.

### **3.3. Vấn đề Cân bằng Giữa Bias và Variance (Bias-Variance Tradeoff)**
- **Underfitting (Thiên vị cao)**: Xảy ra khi mô hình quá đơn giản (bậc $d$ quá thấp) và không thể nắm bắt được xu hướng của dữ liệu.
- **Overfitting (Phương sai cao)**: Xảy ra khi mô hình quá phức tạp (bậc $d$ quá cao). Nó "học thuộc lòng" cả nhiễu trong tập huấn luyện, dẫn đến hiệu suất kém trên dữ liệu mới.
- **Lựa chọn bậc tối ưu**: Để tìm ra bậc $d$ tốt nhất, chúng ta thường sử dụng kỹ thuật **Cross-Validation**.

### **3.4. Ví dụ và Code minh họa**
**Ví dụ thực tế**: Dự đoán tốc độ xe dựa trên thời gian. Dữ liệu có dạng hình parabol. Một mô hình đa thức bậc 2, ví dụ $y = 5 + 15x - 2x^2$, sẽ phù hợp hơn nhiều so với một đường thẳng.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424234272943898664/WMHAfS8LOMQAAAABJRU5ErkJggg.png?ex=68e33507&is=68e1e387&hm=995da0e2cd28284cc35b32bc575cb2bd437a65966a2b94c587c0813ef99126c8&=&format=webp&quality=lossless&width=708&height=541)

---

## **4. Điều chuẩn hóa (Regularization)**

### **4.1. Mục đích**
Khi một mô hình quá phức tạp, nó có xu hướng bị overfitting. Regularization là một kỹ thuật được sử dụng để **kiểm soát độ phức tạp của mô hình** bằng cách thêm một "thành phần phạt" (penalty term) vào hàm mất mát để phạt các trọng số ($w_i$) có giá trị lớn.

### **4.2. Các loại Regularization phổ biến**
- **Lasso Regression (L1 Regularization)**: Phạt tổng giá trị tuyệt đối của các trọng số. Có khả năng đưa một số trọng số về chính xác bằng 0, hữu ích cho việc **lựa chọn feature tự động**.
$$
\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} |w_i|
$$

- **Ridge Regression (L2 Regularization)**: Phạt tổng bình phương của các trọng số. Hiệu quả trong việc xử lý **đa cộng tuyến** (khi các feature tương quan cao).
$$
\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} w_i^2
$$

- **Elastic Net**: Kết hợp cả L1 và L2, tận dụng ưu điểm của cả hai.
$$
\text{Loss} = \text{MSE} + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2
$$

### **4.3. Tham số điều chỉnh $\lambda$ (alpha)**
- $\lambda$ (thường là `alpha` trong scikit-learn) là một siêu tham số kiểm soát mức độ phạt.
    - $\lambda = 0$: Không có regularization.
    - $\lambda$ lớn: Mức phạt mạnh, có thể dẫn đến **underfitting**.
- Việc chọn giá trị $\lambda$ tối ưu thường được thực hiện thông qua **Cross-Validation**.

### **4.4. Ví dụ và Code minh họa**
**Ví dụ thực tế**: Trong dự đoán giá nhà với 100 features, **Lasso** có thể loại bỏ các feature không quan trọng (gán trọng số bằng 0), trong khi **Ridge** sẽ giảm nhỏ trọng số của các feature ít ảnh hưởng.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# --- 1. Tạo dữ liệu giả lập ---
X, y, w = make_regression(
    n_samples=150, n_features=100, n_informative=10,
    noise=15, coef=True, random_state=42
)

# --- 2. Huấn luyện các mô hình ---
lr = LinearRegression().fit(X, y)
ridge = Ridge(alpha=10).fit(X, y)
lasso = Lasso(alpha=1.0).fit(X, y)

# --- 3. Trực quan hóa và so sánh các hệ số (coefficients) ---
plt.figure(figsize=(14, 8))
plt.title('So sánh hệ số của các mô hình', fontsize=16)
plt.plot(w, alpha=0.7, linestyle='none', marker='o', markersize=7, color='red', label='Hệ số thực tế (Ground Truth)')
plt.plot(lr.coef_, alpha=0.6, linestyle='none', marker='s', markersize=7, color='blue', label='Linear Regression')
plt.plot(ridge.coef_, alpha=0.8, linestyle='none', marker='^', markersize=7, color='green', label='Ridge (L2)')
plt.plot(lasso.coef_, alpha=0.9, linestyle='none', marker='x', markersize=7, color='purple', label='Lasso (L1)')
plt.xlabel('Chỉ số của Feature', fontsize=12)
plt.ylabel('Giá trị của hệ số (Weight)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424237480856518748/rs455xxlNpuV2WxWGRkZauPGjaqoqGjE6xRCCCFEYGhKBegWlRBCCCGEEEIIIYSYMmRPISGEEEIIIYQQQogZSAaFhBBCCCGEEEIIIWYgGRQSQgghhBBCCCGEmIFkUEgIIYQQQgghhBBiBpJBISGEEEIIIYQQQogZSAaFhBBCCCGEEEIIIWYgGRQSQgghhBBCCCGEmIFkUEgIIYQQQgghhBBiBpJBISGEEEIIIYQQQogZSAaFhBBCCCGEEEIIIWYgGRQSQgghhBBCCCGEmIFkUEgIIYQQQgghhBBiBpJBISGEEEIIIYQQQogZ6P8HrAUosrtnbisAAAAASUVORK5CYII.png?ex=68e33804&is=68e1e684&hm=2a19aba7c6ebbb196bc2be7fd7b49f50bd36a2f201fb10113c36a757e6a4b654&=&format=webp&quality=lossless&width=1240&height=760)

---

## **5. Đánh giá Mô hình Hồi quy**

### **5.1. Tầm quan trọng của việc đánh giá**
Để biết mô hình hoạt động tốt đến đâu, chúng ta cần các chỉ số (metrics) để đo lường hiệu suất. Luôn luôn đánh giá mô hình trên **tập kiểm thử (test set)**, là dữ liệu mà mô hình chưa từng thấy trong quá trình huấn luyện để tránh kết quả tốt giả tạo do overfitting.

### **5.2. Các chỉ số đánh giá phổ biến**
- **Mean Absolute Error (MAE)**: Trung bình của giá trị tuyệt đối của sai số. Dễ diễn giải và ít bị ảnh hưởng bởi các giá trị ngoại lai (outliers).
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- **Mean Squared Error (MSE)**: Trung bình của bình phương sai số. Phạt các lỗi lớn nặng hơn MAE.
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- **Root Mean Squared Error (RMSE)**: Là căn bậc hai của MSE, có cùng đơn vị với biến target, là chỉ số được sử dụng rộng rãi nhất.
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- **Hệ số xác định R-squared ($R^2$)**: Đo lường tỷ lệ phần trăm phương sai của biến target được giải thích bởi mô hình. $R^2$ có giá trị từ -∞ đến 1.
$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$
- Trong đó, $\bar{y}$ là giá trị trung bình của tất cả các giá trị $y$ thực tế.

### **5.3. Kiểm định chéo (Cross-Validation) **
**Mục đích**: Thay vì chỉ đánh giá trên một lần chia train/test duy nhất (có thể may rủi), kiểm định chéo cho phép chúng ta đánh giá mô hình trên nhiều tập con khác nhau của dữ liệu, mang lại một ước tính **ổn định và đáng tin cậy hơn** về hiệu suất thực sự của mô hình.

**K-Fold Cross-Validation hoạt động như thế nào?**:
1.  **Chia dữ liệu (Split)**: Xáo trộn và chia toàn bộ dữ liệu thành **K** "phần" (folds) bằng nhau (ví dụ K=5 hoặc K=10).
2.  **Lặp và Huấn luyện (Iterate and Train)**: Lặp K lần. Trong mỗi lần lặp, một phần được dùng làm **tập kiểm thử (validation set)**, và K-1 phần còn lại được dùng làm **tập huấn luyện (training set)**.
3.  **Tổng hợp kết quả (Aggregate)**: Hiệu suất cuối cùng của mô hình được tính bằng cách lấy **trung bình** của K kết quả đánh giá thu được.

**Ứng dụng quan trọng nhất: Tinh chỉnh siêu tham số (Hyperparameter Tuning)**
Cross-Validation là công cụ thiết yếu để tìm ra các siêu tham số tốt nhất (ví dụ: bậc `degree` trong Hồi quy Đa thức, `alpha` trong Ridge/Lasso) một cách khách quan. Quy trình là thử các giá trị siêu tham số khác nhau, với mỗi giá trị, ta chạy K-Fold Cross-Validation và chọn giá trị nào cho kết quả trung bình tốt nhất.

### **5.4. Ví dụ và Code minh họa**
**Ví dụ thực tế**: Với mô hình dự đoán giá nhà, **RMSE = 15.000 USD** có nghĩa là trung bình, dự đoán của mô hình sai lệch khoảng 15.000 USD. **$R^2 = 0.85$** có nghĩa là mô hình giải thích được 85% sự biến động về giá nhà.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. Tạo dữ liệu giả lập ---
np.random.seed(42)
dien_tich = np.random.rand(100, 1) * 100 + 50
gia_thuc_te = 50 + 3.5 * dien_tich + np.random.randn(100, 1) * 40

# --- 2. Huấn luyện mô hình ---
model = LinearRegression()
model.fit(dien_tich, gia_thuc_te)

# --- 3. Dự đoán ---
gia_du_doan = model.predict(dien_tich)

# --- 4. Tính toán các chỉ số đánh giá ---
mse = mean_squared_error(gia_thuc_te, gia_du_doan)
rmse = np.sqrt(mse)
r2 = r2_score(gia_thuc_te, gia_du_doan)

# --- Trực quan hóa kết quả ---
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

![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424238155996856461/jB0rBcuvWbIgAAAABJRU5ErkJggg.png?ex=68e338a5&is=68e1e725&hm=6b123f5fa2a008ce262738b339be49a9fa77f1522558846e918fa7b3f243b33c&=&format=webp&quality=lossless&width=1069&height=694)

---

## **6. Tổng kết và Hướng dẫn lựa chọn mô hình**

| Mô hình | Khi nào nên sử dụng? | Ưu điểm | Nhược điểm |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | - Bước khởi đầu cho mọi bài toán hồi quy.<br>- Khi bạn tin rằng có mối quan hệ tuyến tính. | - Nhanh, đơn giản, dễ hiểu.<br>- Không có siêu tham số để tinh chỉnh. | - Không linh hoạt.<br>- Nhạy cảm với các giả định. |
| **Polynomial Regression** | - Khi dữ liệu có xu hướng phi tuyến tính (cong). | - Có thể mô hình hóa các mối quan hệ phức tạp hơn. | - Dễ bị overfitting nếu bậc quá cao.<br>- Cần phải chọn bậc (`degree`). |
| **Ridge Regression (L2)** | - Khi có đa cộng tuyến (các feature tương quan cao). | - Giảm overfitting, ổn định hơn.<br>- Giữ lại tất cả các feature. | - Không thực hiện lựa chọn feature. |
| **Lasso Regression (L1)** | - Khi bạn nghi ngờ nhiều feature không cần thiết. | - Tự động thực hiện feature selection.<br>- Tạo ra mô hình thưa (sparse), dễ diễn giải. | - Có thể loại bỏ các feature hữu ích một cách ngẫu nhiên nếu chúng tương quan. |
| **Elastic Net** | - Khi có đa cộng tuyến và bạn cũng muốn lựa chọn feature. | - Kết hợp sức mạnh của Ridge và Lasso. | - Có hai siêu tham số cần tinh chỉnh. |

**PS**:
- Luôn bắt đầu với mô hình đơn giản nhất (Linear Regression) làm đường cơ sở (baseline).
- Trực quan hóa dữ liệu của bạn để hiểu rõ hơn về mối quan hệ giữa các biến.
- Sử dụng **Cross-Validation** để lựa chọn mô hình và tinh chỉnh siêu tham số một cách đáng tin cậy.
- Đánh giá mô hình bằng nhiều chỉ số khác nhau để có cái nhìn toàn diện về hiệu suất.
