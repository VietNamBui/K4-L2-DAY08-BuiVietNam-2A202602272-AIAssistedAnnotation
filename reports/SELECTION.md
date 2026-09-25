# Vì sao chọn lô này?

## 1. Đề xuất Top 5 frame ưu tiên (với ngân sách giả định chỉ gán 5 ảnh)
Dựa trên 50 dòng đầu của `outputs/selection_round1.csv`, nếu chỉ có ngân sách rà 5 ảnh, tôi đề xuất 5 frame sau nhằm tối ưu hóa độ bất định và tính đa dạng theo thời gian, tránh trùng lặp cảnh:

1. **`frame_0182.jpg`** (Rank 1 | t = 72.8s | Score = 0.9591 | U = 0.9182 | A = 1.0000 | D = 1.0000): Điểm tổng hợp cao nhất toàn bộ pool, chứa 28 box phát hiện với 18 box nằm trong vùng mơ hồ (`n_ambiguous`), đại diện cho đoạn giữa video khi mật độ xe bắt đầu tăng.
2. **`frame_0369.jpg`** (Rank 2 | t = 147.6s | Score = 0.9324 | U = 0.9315 | A = 0.8889 | D = 1.0000): Chứa tới 43 box phát hiện với độ bất định rất cao (U = 0.9315), đại diện cho đoạn cuối video có lưu lượng giao thông đông đúc và nhiều xe ở xa.
3. **`frame_0099.jpg`** (Rank 8 | t = 39.6s | Score = 0.9063 | U = 0.9460 | A = 0.7778 | D = 1.0000): Độ bất định U thuộc nhóm cao nhất (0.9460), đại diện cho đoạn đầu video (t = 39.6s) với đặc trưng xe chạy ngược chiều đèn pha chiếu thẳng vào camera gây chói lóa.
4. **`frame_0227.jpg`** (Rank 11 | t = 90.8s | Score = 0.8915 | U = 0.9164 | A = 0.7778 | D = 1.0000): Thời điểm t = 90.8s cách xa frame_0182 (cách 18 giây) và frame_0369 (cách 56.8 giây), đảm bảo đa dạng hóa bối cảnh ánh sáng và góc quay giữa video, có 37 box với 14 box mơ hồ.
5. **`frame_0326.jpg`** (Rank 4 | t = 130.4s | Score = 0.9155 | U = 0.9310 | A = 0.8333 | D = 1.0000): Mật độ xe cao (39 box, 15 ambiguous), nằm ở t = 130.4s (cách frame_0369 hơn 17 giây nên không bị trùng lặp cảnh).

*Quyết định xử lý ảnh gần trùng:* Loại bỏ các frame có điểm rất cao như `frame_0380.jpg` (Rank 3, t = 152.0s) và `frame_0331.jpg` (Rank 5, t = 132.4s) vì chúng nằm quá gần thời điểm của `frame_0369.jpg` (147.6s) và `frame_0326.jpg` (130.4s). Nếu đưa vào với ngân sách hẹp 5 ảnh, ta sẽ lãng phí chi phí gán nhãn cho các xe gần như giữ nguyên vị trí.

## 2. Phân tích ba frame thuộc lô 12 ảnh model chọn
1. **`frame_0182.jpg`** (Rank 1, Score = 0.9591, t = 72.8s): Đứng đầu danh sách lựa chọn với $A = 1.0$ (18/18 box trong khoảng conf 0.25–0.60). Ảnh contact sheet cho thấy cụm xe ở làn giữa bị ánh sáng đèn đường rọi không đều, nhiều xe con màu tối bị chìm vào nền đường.
2. **`frame_0331.jpg`** (Rank 5, Score = 0.9154, t = 132.4s): Có số lượng box phát hiện lớn nhất trong lô (47 box, 18 ambiguous). Contact sheet thể hiện cảnh cao tốc có nhiều xe tải và xe khách di chuyển song song, tạo ra các vùng che khuất lẫn nhau (occlusion).
3. **`frame_0099.jpg`** (Rank 8, Score = 0.9063, t = 39.6s): Điểm bất định $U = 0.9460$ rất cao. Mô hình gặp khó khăn lớn khi phân biệt thân xe tối màu với vệt sáng đèn pha phản chiếu trên mặt đường ẩm.

## 3. Phân tích frame có điểm cao nhưng không chọn và frame điểm thấp nên xem
* **Frame có điểm cao nhưng không chọn:** `frame_0372.jpg` (Rank 6 | Score = 0.9101 | t = 148.8s). Mặc dù điểm số của frame này cao hơn cả `frame_0312.jpg` (Rank 7, Score = 0.9100) và `frame_0099.jpg` (Rank 8, Score = 0.9063), nó bị thuật toán Active Learning loại bỏ thẳng tay vì thời điểm t = 148.8s chỉ cách `frame_0369.jpg` (Rank 2, t = 147.6s) đúng **1.2 giây** (< `MIN_GAP_S = 2.0s`). Việc loại bỏ này là hoàn toàn chính xác để tránh tốn công rà hai khung hình gần như giống hệt nhau về góc nhìn và vị trí xe.
* **Frame điểm thấp nhưng vẫn nên xem xét:** `frame_0002.jpg` (Rank 18 | Score = 0.8658 | t = 0.8s) hoặc các frame rỗng/rất ít xe: Mô hình ban đầu có thể tự tin giả tạo (overconfident) và dự đoán 0 box hoặc box conf thấp trên các ảnh có ít xe, khiến điểm bất định bị tính thấp, nhưng đây lại là các ca biên (edge case) giúp mô hình học cách không kích hoạt nhầm (giảm False Positive).

## 4. Điều phép chọn này chưa chứng minh về chất lượng mô hình
Điểm bất định (Uncertainty Score) kết hợp độ đa dạng chỉ phản ánh **trạng thái phân vân và thiếu tự tin của mô hình hiện tại** trên dữ liệu pool. Nó **không bảo đảm** rằng việc gán nhãn các ảnh này sẽ tự động giúp mô hình tăng chỉ số AP50 trên tập kiểm thử (test set). Cụ thể:
- Nếu các ảnh được chọn chứa quá nhiều ca nhiễu cá biệt (xe quá tối, ánh sáng phản chiếu kỳ dị không lặp lại trong test set), mô hình có thể bị overfitting vào 12 ảnh này.
- Tập test cố định (20 ảnh) có phân phối riêng; việc cải thiện trên 12 ảnh khó ở pool có thể khiến mô hình dịch chuyển ngưỡng quyết định (precision/recall trade-off) làm điểm AP50 trên tập test tham chiếu thay đổi mà không nhất thiết phản ánh chất lượng suy luận tổng quát.
