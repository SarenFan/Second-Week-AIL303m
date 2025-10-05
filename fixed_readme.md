# **Ghi chú cá nhân**
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

![Alt text](https://media.discordapp.net/attachments/1056943939464212542/1424231192244523108/9BAIBbkhARERERGmJ81kiotRi0ZaIiIiIiIiIiIiIhIQ7gRGREREREREREREREpCEs2hIRERERERERERERBrCoi0RERERERERERERGRhrBoS0RERERERERERERKQhLNoSERERERERERERaQiLtkREREREREREREQa8v8B8sxTSsfKnPUAAAAASUVORK5CYII.png?ex=68e33229&is=68e1e0a9&hm=4778f3b7167fda714de35d7fe5291967de07c23fa5f5fb2a9cfb2c624a8f5654&=&format=webp&quality=lossless&width=1730&height=738)

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

$$y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n + \epsilon$$

- $y$: Biến mục tiêu (target) - giá trị chúng ta muốn dự đoán.
- $x_1, x_2, \dots, x_n$: Các biến độc lập (features) - thông tin đầu vào.
- $w_0$: Hệ số chặn (Intercept hoặc Bias) - là giá trị của $y$ khi tất cả các feature $x_i$ bằng 0.
- $w_1, w_2, \dots, w_n$: Các hệ số hồi quy (Coefficients hoặc Weights) - đại diện cho độ dốc. Mỗi $w_i$ cho biết mức độ ảnh hưởng của feature $x_i$ lên $y$. Nếu $w_i$ dương, $y$ tăng khi $x_i$ tăng, và ngược lại.
- $\epsilon$: Sai số ngẫu nhiên (Residuals) - phần biến thiên của $y$ không thể giải thích được bởi mô hình. Giả định rằng sai số này tuân theo phân phối chuẩn với kỳ vọng bằng 0.

### **2.3. Hàm mất mát (Loss Function) và Tối ưu hóa**
- **Mục tiêu**: Chúng ta cần một cách để đo lường "mức độ sai" của mô hình. Đây là lúc hàm mất mát phát huy tác dụng.
- **Mean Squared Error (MSE)**: Là hàm mất mát phổ biến nhất cho hồi quy. Nó tính trung bình của tổng các bình phương sai số giữa giá trị thực tế và giá trị dự đoán.
  
  $$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
  
  - **Tại sao lại bình phương?** Việc bình phương giúp loại bỏ dấu âm của sai số và "trừng phạt" các lỗi lớn nặng hơn nhiều so với các lỗi nhỏ.
- **Phương pháp tối ưu hóa (Tìm w tốt nhất)**:
    - **Ordinary Least Squares (OLS)**: Một phương pháp giải tích cho ra nghiệm chính xác (closed-form solution). Nó hoạt động hiệu quả với các bộ dữ liệu không quá lớn.
    - **Gradient Descent**: Một thuật toán lặp. Nó bắt đầu với các trọng số $w$ ngẫu nhiên, sau đó tính đạo hàm (gradient) của hàm mất mát theo từng trọng số và cập nhật các trọng số theo hướng ngược với gradient để giảm thiểu hàm mất mát.
    
    $$w_j := w_j - \alpha \frac{\partial}{\partial w_j} MSE$$
    
    - $\alpha$ (Learning rate - Tốc độ học): Là một siêu tham số (hyperparameter) quyết định "bước đi" lớn hay nhỏ trong mỗi lần cập nhật. Learning rate quá nhỏ sẽ khiến việc hội tụ rất chậm, trong khi learning rate quá lớn có thể khiến thuật toán "vượt" qua điểm tối ưu.
    - **Các biến thể**: Stochastic Gradient Descent (SGD) cập nhật trọng số sau mỗi mẫu dữ liệu, Mini-batch Gradient Descent cập nhật sau một nhóm nhỏ dữ liệu. Các biến thể này giúp tăng tốc độ hội tụ trên các bộ dữ liệu lớn.

### **2.4. Ví dụ và Code minh họa**
**Ví dụ thực tế**: Dự đoán mức lương dựa trên số năm kinh nghiệm.
- **Dữ liệu**: Kinh nghiệm (x) = [1, 2, 3, 4] năm, Lương (y) = [30k, 40k, 50k, 60k] USD.
- **Mô hình**: Sau khi huấn luyện, mô hình có thể tìm ra mối quan hệ là $y = 20k + 10k \times x$.
- **Dự đoán**: Với một người có 5 năm kinh nghiệm, mô hình sẽ dự đoán lương là $20k + 10k \times 5 = 70k$ USD.
- Nếu thêm một feature nữa như "bằng cấp" (được mã hóa thành số), mô hình sẽ trở thành Multiple Linear Regression.

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
Khi mối quan hệ giữa feature và target không phải là một đường thẳng mà là một đường cong, Linear Regression sẽ hoạt động kém hiệu quả. Polynomial Regression giải quyết vấn đề này bằng cách tạo ra các feature mới là lũy thừa của các feature ban đầu, sau đó áp dụng mô hình Linear Regression trên tập feature đã được mở rộng này.
- **Bản chất**: Về cơ bản, đây vẫn là một mô hình hồi quy tuyến tính, nhưng nó tuyến tính trên một không gian feature đã được biến đổi (feature engineering) thay vì không gian feature ban đầu.

### **3.2. Công thức**
Mô hình hồi quy đa thức bậc $n$ với một feature $x$ có dạng:

$$y = w_0 + w_1x + w_2x^2 + \dots + w_nx^n + \epsilon$$

- Để huấn luyện mô hình này, chúng ta chỉ cần biến đổi feature $x$ ban đầu thành một tập hợp các feature mới là $[x, x^2, x^3, \dots, x^n]$ và sau đó áp dụng thuật toán Linear Regression thông thường.

### **3.3. Vấn đề Cân bằng Giữa Bias và Variance (Bias-Variance Tradeoff)**
Đây là một khái niệm cốt lõi khi làm việc với Polynomial Regression.
- **Underfitting (Thiên vị cao - High Bias)**: Xảy ra khi mô hình quá đơn giản (ví dụ, bậc đa thức quá thấp) và không thể nắm bắt được xu hướng phức tạp của dữ liệu. Mô hình sẽ có hiệu suất kém trên cả tập huấn luyện và tập kiểm thử.
- **Overfitting (Phương sai cao - High Variance)**: Xảy ra khi mô hình quá phức tạp (bậc đa thức quá cao). Nó "học thuộc lòng" cả nhiễu (noise) trong tập huấn luyện, dẫn đến hiệu suất rất tốt trên tập huấn luyện nhưng cực kỳ kém trên dữ liệu mới (tập kiểm thử).
- **Lựa chọn bậc (degree) tối ưu**: Để tìm ra bậc đa thức tốt nhất, chúng ta thường sử dụng kỹ thuật **Cross-Validation**. Kỹ thuật này giúp đánh giá hiệu suất của mô hình trên nhiều tập dữ liệu con khác nhau để có một ước tính đáng tin cậy hơn về khả năng tổng quát hóa của nó.

### **3.4. Ví dụ và Code minh họa**
**Ví dụ thực tế**: Dự đoán tốc độ của một chiếc xe dựa trên thời gian, khi xe tăng tốc rồi giảm tốc.
- **Dữ liệu**: Thời gian (x) = [1, 2, 3, 4, 5], Tốc độ (y) = [10, 20, 25, 20, 10] km/h. Dữ liệu này rõ ràng có dạng hình parabol.
- **Mô hình**: Một mô hình đa thức bậc 2, ví dụ $y = 5 + 15x - 2x^2$, sẽ phù hợp hơn nhiều so với một đường thẳng.
- **Dự đoán**: Với x=6, mô hình có thể dự đoán tốc độ đang giảm, ví dụ y=3 km/h.

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
Khi một mô hình quá phức tạp (ví dụ có quá nhiều features hoặc bậc đa thức cao), nó có xu hướng bị overfitting. Regularization là một kỹ thuật được sử dụng để **kiểm soát độ phức tạp của mô hình** và **ngăn chặn overfitting**.
- **Cách hoạt động**: Nó thêm một "thành phần phạt" (penalty term) vào hàm mất mát (loss function). Thành phần này sẽ phạt các trọng số (coefficients) có giá trị lớn. Bằng cách khuyến khích các trọng số nhỏ hơn, regularization làm cho mô hình trở nên đơn giản hơn và ít nhạy cảm hơn với nhiễu trong dữ liệu huấn luyện, từ đó cải thiện khả năng tổng quát hóa.

### **4.2. Các loại Regularization phổ biến**
- **Lasso Regression (L1 Regularization)**: Thêm vào hàm mất mát tổng các giá trị **tuyệt đối** của các trọng số.
  
  $$Loss = MSE + \lambda \sum_{i=1}^n |w_i|$$
  
  - **Đặc điểm chính**: Lasso có khả năng đưa một số trọng số về **chính xác bằng 0**. Điều này biến nó thành một công cụ hữu ích cho việc **lựa chọn feature tự động (feature selection)**, vì các feature có trọng số bằng 0 sẽ bị loại bỏ khỏi mô hình.
  - **Khi nào nên dùng**: Khi bạn nghi ngờ rằng có nhiều feature không quan trọng hoặc dư thừa trong bộ dữ liệu của mình.

- **Ridge Regression (L2 Regularization)**: Thêm vào hàm mất mát tổng các **bình phương** của các trọng số.
  
  $$Loss = MSE + \lambda \sum_{i=1}^n w_i^2$$
  
  - **Đặc điểm chính**: Ridge làm cho các trọng số nhỏ lại, tiến gần về 0, nhưng **hiếm khi bằng 0**. Nó rất hiệu quả trong việc xử lý **đa cộng tuyến (multicollinearity)**, tức là khi các feature có tương quan cao với nhau.
  - **Khi nào nên dùng**: Khi bạn tin rằng tất cả các feature đều có thể có ích và muốn giảm tác động của các feature ít quan trọng hơn.

- **Elastic Net**: Là sự kết hợp của cả L1 và L2.
  
  $$Loss = MSE + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2$$
  
  - **Đặc điểm chính**: Tận dụng được ưu điểm của cả hai. Nó có thể thực hiện lựa chọn feature (như Lasso) trong khi vẫn xử lý được vấn đề đa cộng tuyến (như Ridge).
  - **Khi nào nên dùng**: Khi bạn có nhiều feature tương quan cao với nhau và đồng thời muốn loại bỏ các feature không cần thiết.

### **4.3. Tham số điều chỉnh $\lambda$ (alpha)**
- $\lambda$ (thường được gọi là `alpha` trong `scikit-learn`) là một siêu tham số kiểm soát mức độ phạt.
    - **$\lambda = 0$**: Không có regularization, mô hình trở thành Linear Regression thông thường.
    - **$\lambda$ lớn**: Mức phạt rất mạnh, các trọng số sẽ bị ép về gần 0, có thể dẫn đến **underfitting**.
    - **$\lambda$ nhỏ**: Mức phạt yếu, mô hình có thể vẫn bị overfitting.
- Việc chọn giá trị $\lambda$ tối ưu thường được thực hiện thông qua **Cross-Validation**.

### **4.4. Ví dụ và Code minh họa**
**Ví dụ thực tế**: Trong một bài toán dự đoán giá nhà với 100 features (bao gồm diện tích, số phòng, tuổi của ngôi nhà, màu sơn tường, khoảng cách đến công viên,...).
- **Lasso** có thể tự động xác định rằng "màu sơn tường" là một feature không quan trọng và gán cho nó trọng số bằng 0, loại bỏ nó khỏi mô hình.
- **Ridge** sẽ giữ lại tất cả 100 features nhưng sẽ giảm nhỏ trọng số của các feature ít ảnh hưởng như "khoảng cách đến công viên" nếu nó không quá quan trọng.

```python
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
# Alpha là tham số điều chỉnh độ mạnh của regularization. Alpha cà