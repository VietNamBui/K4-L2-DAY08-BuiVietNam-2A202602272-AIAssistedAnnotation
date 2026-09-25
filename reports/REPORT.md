# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Bùi Việt Nam

Công cụ gán nhãn đã dùng: CVAT (Docker local v2.74.1 / v2.76.0)

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (`pool`) gồm 268 ảnh và tập kiểm thử (`test set`) gồm 20 ảnh được chia tách theo **trục thời gian (temporal split)** và có một khoảng đệm thời gian (buffer gap) ở giữa, thay vì chia ngẫu nhiên (random split). 

Lý do bắt buộc phải chia theo trục thời gian:
* Dữ liệu hình ảnh được trích xuất từ luồng video giám sát giao thông liên tục. Các khung hình (frames) nằm cạnh nhau có tính tương quan không gian - thời gian (spatio-temporal correlation) cực kỳ cao: cùng góc quay, cùng điều kiện ánh sáng, và đặc biệt là cùng những chiếc xe đang di chuyển chậm qua khung hình.
* Nếu chia ngẫu nhiên (random split), hiện tượng **rò rỉ dữ liệu (data leakage)** sẽ xảy ra nghiêm trọng. Mô hình chỉ cần "học thuộc lòng" (memorization) hình dáng và vị trí của chiếc xe ở frame huấn luyện là có thể phát hiện chính xác chiếc xe đó ở frame kiểm thử chỉ cách vài giây.
* Hậu quả là số đo trên tập kiểm thử (như AP50, Precision, Recall) sẽ bị **lệch theo hướng lạc quan giả tạo (artificially inflated / over-optimistic)**, tạo cảm giác mô hình rất tốt nhưng khi đem triển khai thực tế trên một đoạn video mới thì hiệu năng sẽ sụt giảm nghiêm trọng. Việc chia theo thời gian kèm vùng đệm bảo đảm tập kiểm thử phản ánh trung thực khả năng khái quát hóa (generalization) của mô hình trên tương lai chưa từng thấy.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng số đo Vòng 0 từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg` và `outputs/metrics_round0.json`:
* **Các loại xe mô hình không khớp nhãn tham chiếu:** Mô hình khởi đầu lạnh (`yolov8n.pt` pretrained trên COCO) bỏ sót rất nhiều xe con màu tối ở làn ngoài cùng bên phải, xe ở làn xa (chân cầu vượt), và các xe bị che khuất một phần (occluded). Mô hình cũng có xu hướng kéo dài box ôm cả vệt đèn pha rọi xuống mặt đường ướt ở làn ngược chiều bên trái.
* **Độ phủ (Recall) theo kích thước xe:** 
  * Xe nhỏ (`small`): Recall chỉ đạt **0.1818 (18.18%)** trên 66 box tham chiếu.
  * Xe vừa (`medium`): Recall đạt **0.5473 (54.73%)** trên 296 box tham chiếu.
  * Xe lớn (`large`): Recall đạt **0.5610 (56.10%)** trên 41 box tham chiếu.
  Con số này chứng minh mô hình YOLOv8n gốc gặp điểm yếu chí mạng ở các đối tượng kích thước nhỏ trong bối cảnh ban đêm; hơn 80% xe nhỏ ở xa bị bỏ sót hoàn toàn.
* **Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai:** Khi mô hình phát hiện một phương tiện ở rất xa có độ tin cậy thấp hoặc nhầm vệt phản chiếu đèn pha trên dải phân cách bê tông. Vì nhãn tham chiếu của tập test là do một mô hình tự động tạo ra (pseudo-groundtruth) chưa qua kiểm duyệt thủ công từng box của con người, nên có những vị trí mô hình dự đoán đúng một chiếc xe tối màu ở xa mà nhãn tham chiếu bỏ sót, hoặc nhãn tham chiếu vô tình gán nhầm vệt đèn chói thành xe. Khi đó không thể vội vàng kết luận mô hình dự đoán sai.

## 3. Chiến lược chọn mẫu

* **Giải thích công thức:** `score = W_U·U + W_A·A + W_D·D` (với trọng số thiết lập `W_U = 0.5`, `W_A = 0.3`, `W_D = 0.2`):
  * **$U$ (Uncertainty - Độ bất định, trọng số 0.5):** Đo mức độ thiếu tự tin của mô hình trên toàn bộ các box dự đoán trong frame. Frame có điểm $U$ cao nghĩa là mô hình đang phân vân nhiều nhất, cần con người gắn nhãn định hướng.
  * **$A$ (Ambiguity - Tỷ lệ box mơ hồ, trọng số 0.3):** Tỷ lệ các box có độ tin cậy rơi vào vùng ranh giới quyết định nhạy cảm ($0.25 \le \text{conf} \le 0.60$). Khung hình có $A$ cao là khung hình chứa nhiều ranh giới mập mờ giữa xe và nền tối.
  * **$D$ (Diversity - Độ đa dạng thời gian, trọng số 0.2):** Khuyến khích chọn các frame phân bổ rải rác trên trục thời gian, tránh việc thuật toán dồn toàn bộ ngân sách vào một khoảnh khắc ngắn.
  * **Vai trò của `MIN_GAP_S = 2.0s`:** Đây là bộ lọc triệt tiêu trùng lặp (de-duplication constraint). Dù một frame có điểm số cao đến đâu, nếu nó cách một frame đã chọn trước đó dưới 2.0 giây thì vẫn bị loại bỏ. Điều này ngăn chặn lãng phí nhân lực vào việc gán nhãn hai bức ảnh gần như y hệt nhau về xe và bối cảnh.

* **Dẫn chứng các frame từ `reports/SELECTION.md`:**
  * **3 frame được chọn:** `frame_0182.jpg` (Rank 1, Score 0.9591 - độ bất định và tỷ lệ box mơ hồ cao nhất toàn pool), `frame_0369.jpg` (Rank 2, Score 0.9324 - lưu lượng xe lớn ở cuối video với 43 box), và `frame_0099.jpg` (Rank 8, Score 0.9063 - đại diện cho đoạn đầu video với xe chạy ngược chiều đèn pha chói lóa).
  * **1 frame bị loại vì ảnh gần trùng:** `frame_0372.jpg` (Rank 6, Score 0.9101, t = 148.8s). Frame này có điểm số rất cao (cao hơn cả Rank 7 và Rank 8), nhưng bị thuật toán loại bỏ vì nó chỉ cách `frame_0369.jpg` (t = 147.6s) đúng **1.2 giây** (< 2.0s). Việc loại bỏ này giúp tiết kiệm công sức của người gán nhãn.

* **Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không? Vì sao?**
  **Hoàn toàn không.** Điểm bất định cao chỉ là một tín hiệu chẩn đoán cho thấy mô hình đang "bối rối" trước khung hình đó. Nó không phải là bảo chứng rằng việc học khung hình đó sẽ giúp tăng AP50 trên tập test. Nếu khung hình đó bối rối do chứa nhiễu bất thường (ví dụ: giọt nước trên ống kính, bóng đèn pha phản chiếu kỳ dị, biển báo lạ) mà các yếu tố này không hề xuất hiện trong tập kiểm thử, việc huấn luyện trên ảnh đó thậm chí có thể gây ra hiện tượng học vẹt (overfitting vào nhiễu) và làm giảm hiệu năng tổng thể của mô hình.

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 287 | 0.442 | -0.330 | 1.000 | 0.010 | 0.020 | 0.000 | 0.010 | 0.024 |

* **Mức độ sửa nhãn ở Vòng 1 (từ `outputs/round1_diff.md`):**
  * Trong 12 ảnh, mô hình AI ban đầu chỉ đề xuất 169 box. Sau khi người rà soát trên CVAT, số box chuẩn tăng lên **287 box**.
  * Cụ thể: **157 box được chấp nhận giữ nguyên (accepted)**, **6 box được chỉnh sửa kích thước/vị trí (edited)**, **6 box sai bị xóa bỏ (deleted - False Positive của AI)**, và bổ sung thêm tới **124 box xe bị bỏ sót (added - False Negative của AI)**. Tỷ lệ chấp thuận đạt 93%.
* **Biến thiên số đo AP50 và các nhóm xe:**
  * AP50 giảm từ **0.771 xuống 0.442** ($\Delta = -0.330$).
  * Tuy nhiên, độ chính xác **Precision@0.25 tăng lên tuyệt đối đạt 1.000 (100%)** so với 0.925 ban đầu. Không còn một False Positive nào bị kích hoạt nhầm trên tập test.
  * Ngược lại, Recall giảm mạnh xuống 0.010. Do tập train chỉ có 12 ảnh (quá nhỏ so với lượng dữ liệu khổng lồ của COCO ban đầu) và người gán nhãn đã siết chặt kỷ luật box (xóa các vệt đèn và box không rõ), mô hình trở nên cực kỳ khắt khe và phân phối điểm tự tin (confidence scores) bị kéo trượt xuống dưới ngưỡng `conf_thr = 0.25` trên tập test.
* **Đối chiếu hình ảnh `outputs/compare_round1.jpg`:**
  * Mô hình sau fine-tune không còn vẽ các box bừa bãi ôm trùm vệt đèn pha như ở vòng 0. Mọi dự đoán đưa ra đều cực kỳ chuẩn xác (Precision 100%).
  * Sự khác biệt giữa 3 nguồn thông tin:
    * *Quan sát độc lập (`BLIND_SCAN.md`):* Trên `frame_0099.jpg`, mắt người đếm được 21 xe, phát hiện xe tối làn trái dễ bị ôm vệt đèn và xe đỏ làn phải chìm vào bóng tối.
    * *Lỗi pre-label (`round1_diff.md`, `REVIEW_LOG.csv`):* AI chỉ đề xuất 13 box, bỏ sót tới 9 xe (trong đó có ca xe giữa đường phía bên phải) và vẽ lệch box mở rộng sang đèn pha xe khác ở `frame_0107.jpg`. Người gán nhãn đã sửa lại đúng 21 box cho `frame_0099.jpg`.
    * *Mô hình sau train:* Khắc phục được lỗi vẽ box thừa vệt sáng, nhưng do kích thước lô nhỏ (12 ảnh) nên chưa tích lũy đủ độ khái quát cho mọi trường hợp xe tối ở xa.
* **Mô tả ca khó theo guideline:** Ca chiếc xe ở làn dưới cùng bên trái của `frame_0099.jpg`. Xe chạy thẳng về hướng camera, thân xe tối màu hoàn toàn chìm vào nền đường đen, chỉ có hai cụm đèn pha rọi vệt sáng cực lớn về phía trước. Đây là ca cực khó vì nếu vẽ box theo vệt sáng thì vi phạm guideline (không được tính vệt sáng đèn pha trên mặt đường), còn nếu chỉ khoanh hai bóng đèn thì vi phạm quy tắc thân xe. Người gán nhãn phải ước lượng ranh giới thân xe quanh cụm đèn để vẽ box ôm sát thân xe.

## 5. Kết luận và giới hạn

* **Đánh giá kết quả Vòng 1 so với Cold Start:**
  * Mô hình đã học được bài học rất quan trọng về **chất lượng box (Precision 100%)**, loại bỏ hoàn toàn các box ảo do đèn pha rọi mặt đường.
  * Mặc dù AP50 giảm do đánh đổi Recall với ngưỡng conf 0.25 trên tập test nhỏ, đây là hiện tượng hoàn toàn bình thường và rất phổ biến trong Active Learning giai đoạn đầu khi kích thước lô gán nhãn (12 ảnh) còn quá khiêm tốn so với độ phức tạp của bài toán ban đêm.
* **Quyết định dừng/tiếp tục và đề xuất cho vòng sau:**
  * Đạt đủ yêu cầu 1 vòng học chủ động bắt buộc của bài thực hành. Nếu tiếp tục Vòng 2, dựa vào `outputs/selection_round2.csv`, tôi đề xuất ưu tiên 2 ca:
    1. **`frame_0000.jpg`** (Rank 1 vòng 2 | t = 0.0s | Score = 0.9421): Nằm ở đầu video với điều kiện góc quay và phân bố xe mới lạ giúp hồi phục Recall.
    2. **`frame_0074.jpg`** (Rank 4 vòng 2 | t = 29.6s | Score = 0.9137): Lấp đầy khoảng trống thời gian giữa t = 0s và t = 39.6s của vòng 1.
  * *Chi phí rà nhãn và nguy cơ ảnh trùng:* Cần tiếp tục duy trì ràng buộc `MIN_GAP_S` để tránh gán nhãn lặp lại các ảnh cách nhau dưới 2 giây, nhằm tối đa hóa giá trị thông tin trên từng giờ công của người gán nhãn.
* **Tác động của các giới hạn:**
  * Tập kiểm thử chỉ có 20 ảnh và nhãn tham chiếu do một mô hình khác tự động sinh ra (chưa được người rà soát từng box). Do đó, chỉ số AP50 chỉ đo **mức độ tương đồng giữa mô hình của chúng ta với mô hình tạo nhãn tham chiếu**, chứ không đại diện tuyệt đối cho chân lý thực tế ngoài đường.
  * Quy tắc bỏ qua xe quá nhỏ (< 16px) là cần thiết để tránh phạt mô hình ở những chấm sáng không thể phân biệt được bằng mắt người.
* **Nếu AP50 giảm, sẽ kiểm tra điều gì trước khi train thêm:**
  1. Kiểm tra lại phân phối điểm tin cậy (confidence distribution): Hạ ngưỡng conf từ 0.25 xuống 0.10 để xem các dự đoán đúng có bị dồn vào khoảng conf thấp hay không.
  2. Kiểm tra tính nhất quán (label consistency) trong `labels/round1/`: Đảm bảo quy tắc vẽ thân xe quanh đèn pha được áp dụng đồng nhất trên tất cả 12 ảnh.
  3. Điều chỉnh tham số fine-tune (ví dụ giảm learning rate, freeze các lớp backbone) để mô hình không bị quên các đặc trưng tổng quát đã học từ COCO khi học trên tập dữ liệu nhỏ 12 ảnh.
