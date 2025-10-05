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
    - **Phương sai không đổi của sai số (Homoscedasticity)**: Phương sai của sai sốPS**:
- Luôn bắt đầu với mô hình đơn giản nhất (Linear Regression) làm đường cơ sở (baseline).
- Trực quan hóa dữ liệu của bạn để hiểu rõ hơn về mối quan hệ giữa các biến.
- Sử dụng **Cross-Validation** để lựa chọn mô hình và tinh chỉnh siêu tham số một cách đáng tin cậy.
- Đánh giá mô hình bằng nhiều chỉ số khác nhau để có cái nhìn toàn diện về hiệu suất.
