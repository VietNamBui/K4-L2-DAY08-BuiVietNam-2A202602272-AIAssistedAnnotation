# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 13 xe (tính các xe quan sát được thân xe và đèn xe ở hai chiều di chuyển; không tính các vệt sáng/chấm sáng li ti quá xa ở chân cầu vượt phía xa có chiều cao dưới 16 pixel).

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Xe ô tô màu tối ở làn dưới cùng bên trái (chạy về phía camera): Thân xe tối chìm vào nền đường đêm, đèn pha chiếu xuống mặt đường tạo vệt sáng rất lớn. Mô hình AI dễ vẽ box ôm cả vệt đèn pha trên mặt đường (vi phạm guideline) hoặc chỉ khoanh hai cụm đèn mà bỏ sót phần thân xe tối phía sau.
2. Xe ô tô ở làn bên phải di chuyển ra xa (hướng lên dốc cầu, có đèn hậu màu đỏ): Thân xe tối hòa lẫn vào bóng tối giữa các làn xe, độ tương phản rất thấp khiến AI dễ bỏ sót (False Negative), hoặc AI nhận diện nhầm ánh phản chiếu đèn phanh trên mặt đường thành một xe độc lập.
