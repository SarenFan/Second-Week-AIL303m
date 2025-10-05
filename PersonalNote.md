# Personal Note: Lý thuyết Hồi quy trong Học máy có Giám sát 


## 1. Giới thiệu về Hồi quy (Regression)
- **Học máy có giám sát (Supervised Learning)**: Đây là một loại học máy nơi mô hình được huấn luyện trên dữ liệu đã được gắn nhãn, nghĩa là mỗi mẫu dữ liệu đầu vào (features) đi kèm với giá trị đầu ra đúng (labels hoặc targets). Mục tiêu là học cách dự đoán đầu ra cho dữ liệu mới dựa trên các mẫu đã học. Quy trình bao gồm: thu thập dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá, và triển khai. Supervised Learning khác với unsupervised learning (không có nhãn) ở chỗ nó tập trung vào dự đoán chính xác dựa trên dữ liệu đã biết.
  
- **Hồi quy (Regression)**: Là một phần của supervised learning, dùng để dự đoán giá trị liên tục (continuous) thay vì giá trị rời rạc. Regression giả định rằng có mối quan hệ toán học giữa các features và target, và mô hình cố gắng ước lượng mối quan hệ này. Các giả định cơ bản bao gồm: tuyến tính (nếu là linear), độc lập giữa các quan sát, và phân bố chuẩn của lỗi. Nếu vi phạm các giả định này, cần sử dụng các kỹ thuật biến đổi dữ liệu hoặc mô hình khác.

- **So sánh với phân loại (Classification)**: Regression dự đoán số thực (ví dụ: 3.14, 100.5), trong khi classification dự đoán nhãn hạng mục (ví dụ: "có/không", "mèo/chó"). Regression thường sử dụng hàm mất mát như MSE để đo lường sai số liên tục, còn classification dùng cross-entropy cho xác suất.

**Ví dụ thực tế**: Trong dự đoán giá nhà, regression có thể dự đoán giá chính xác là 250.000 USD dựa trên diện tích, số phòng, và vị trí. Ngược lại, classification có thể phân loại nhà là "rẻ" hoặc "đắt" dựa trên ngưỡng giá.

![Alt text](https://www.simplilearn.com/ice9/free_resources_article_thumb/Regression_vs_Classification.jpg)

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

**Hình ảnh minh họa**: Biểu đồ đường thẳng linear regression khớp với dữ liệu.

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

**Hình ảnh minh họa**: Biểu đồ đường cong polynomial regression.

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

**Hình ảnh minh họa**: Bảng so sánh Ridge vs Lasso.

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

**Hình ảnh minh họa**: Biểu đồ các metrics đánh giá.

---

## 6. Tổng kết
- **Linear Regression**: Nền tảng cho dữ liệu tuyến tính, dễ triển khai nhưng giới hạn với phi tuyến.
- **Polynomial Regression**: Linh hoạt hơn cho dữ liệu cong, nhưng cần kiểm soát degree để tránh overfitting.
- **Regularization**: Công cụ mạnh mẽ để cân bằng bias-variance, đặc biệt với dữ liệu cao chiều.
- **Đánh giá**: Luôn dùng nhiều metrics và cross-validation để đảm bảo mô hình robust.

**Lưu ý**: Để hiểu sâu, hãy thử tính tay các công thức trên dữ liệu nhỏ. Nếu cần code ví dụ hoặc biểu đồ tùy chỉnh, hãy cho mình biết!
