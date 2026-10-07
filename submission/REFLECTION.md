# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là sự "nghịch đảo" giữa training loss và hiệu năng thực tế trên bài toán: run `attn_only` (với rank matched $r=283$) có training loss thấp hơn `correct` ($0.5376$ so với $0.6270$), nhưng trên tập kiểm thử target nó không hề vượt trội hơn ($0.970$ so với $0.970$). Điều này chứng minh trực quan rằng training loss là một thước đo thay thế (proxy metric) có thể đánh lừa người làm AI, việc tối ưu loss trên không gian hẹp chỉ là ghi nhớ dữ liệu cục bộ chứ không đồng nghĩa với khả năng suy luận tốt hơn. Ngoài ra, việc QLoRA cắt giảm tới 56% VRAM (từ 8.78 GB xuống 3.86 GB) nhưng phải trả giá bằng việc train chậm hơn 23% và giảm nhẹ điểm target từ 0.97 xuống 0.94 cũng là số đo thực nghiệm rất ấn tượng.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở bước chạy toàn bộ pipeline huấn luyện và đánh giá trên Google Colab (tổng cộng 53.3 phút, trong đó NB4 huấn luyện 3 run đối chứng mất 22.6 phút và NB5 chấm điểm mất 11.7 phút). Ban đầu tôi dự đoán bước tải model 9.32 GB từ HuggingFace sẽ lâu nhất do phụ thuộc đường truyền, nhưng thực tế thời gian sinh văn bản (text generation decode cho 50 mẫu qua 3 baseline và 4 adapter) mới là phần tiêu tốn nhiều thời gian nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng: (1) Cứ tăng rank LoRA càng cao thì mô hình sẽ càng thông minh; (2) Chỉ cần gắn LoRA vào các ma trận attention ($q, v$) là đủ như các hướng dẫn cũ năm 2023–2024; và (3) Fine-tuning hiển nhiên sẽ luôn đánh bại prompting. Giờ đây tôi hiểu rằng rank LoRA phải tương thích với lượng thông tin trong dữ liệu chứ không phải nút vặn chất lượng; việc gắn adapter trải đều toàn bộ các lớp `text-linear` với rank vừa phải ($r=16$) mang lại hiệu quả vượt trội so với dồn rank khổng lồ ($r=283$) vào $q, v$; và một prompt được tối ưu kỹ lưỡng (baseline b) đã có thể giải quyết tới 76.5% bài toán mà không tốn chi phí huấn luyện hay bảo trì checkpoint.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để rà soát mã nguồn lab, kiểm tra các điều kiện trong `scripts/verify.py`, cấu hình môi trường local (Python 3.11, PyTorch CUDA 12.4, fix lỗi xuống dòng CRLF trên Windows) và hỗ trợ giải thích các cơ chế kỹ thuật trong deck bài giảng. Chỗ AI ban đầu chưa tối ưu là dự định chạy thử nghiệm ngay trên máy cá nhân với card GTX 1660 SUPER 6GB — nếu cố chạy model Qwen3.5-4B tại local thì chắc chắn sẽ gặp lỗi tràn bộ nhớ (CUDA Out of Memory), trước khi thống nhất chuyển toàn bộ quá trình train lên Google Colab Free T4 16GB theo đúng thiết kế của tác giả lab.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Nếu ngày mai làm cho một khách hàng thật, bước đầu tiên tôi làm KHÔNG PHẢI là mở notebook lên train ngay, mà là:
1. Đóng băng một tập dữ liệu đánh giá chuẩn (Golden Evaluation Set) và xây dựng một baseline prompt engineering thật tốt (Baseline b) để đo đạc chính xác ngưỡng năng lực hiện tại của mô hình nền.
2. Kiểm tra tính đúng đắn của Chat Template và Loss Mask (bằng cách giải mã ngược token labels như NB1) để đảm bảo 100% rằng prompt không bị rò rỉ vào hàm loss trước khi bấm máy huấn luyện.
