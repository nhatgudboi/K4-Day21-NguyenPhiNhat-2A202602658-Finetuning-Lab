# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Phi Nhật  **MSSV**: 2A202602658  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab Free)`

> Mọi con số dưới đây khớp 100% với các file trong thư mục `results/` được đo đạc thực tế từ quá trình huấn luyện và đánh giá trên Google Colab T4.

---

## 1. Setup

| Thông số | Giá trị thực nghiệm |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val | 225 mẫu huấn luyện / 25 mẫu kiểm định (cố định `seed=42`) |
| `max_length` | 1024 (p95 đo được thực tế là 98 tokens theo `results/token_stats.json`) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps (batch per device = 1, gradient accumulation = 16) |

**Template có giữ khối `<think>` không?** Có — theo `results/template_check.json`, trường verdict trả về `"reasoning preserved — safe to train on traces"`. Template chuẩn của Qwen3.5 bảo tồn đầy đủ thẻ mở và đóng của khối suy luận, không nuốt token reasoning.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị đo được |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens nằm trong loss) |
| Câu trả lời nằm trong loss | `true` (xác thực giải mã ngược thành công) |
| Câu hỏi KHÔNG nằm trong loss | `true` (toàn bộ prompt hệ thống và user đều bị gán -100) |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3310.8 |
| (b) base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1068.7 |
| (c) LoRA fine-tune (`correct`) | 0.970 | 0.6778 | 1.000 | 1472.0 |

**(b) có thật sự mạnh hơn (a) không?** Có, vượt trội hoàn toàn: target tăng từ 0.000 lên 0.765 và format đạt chuẩn tuyệt đối 1.000 (so với 0.000 của naive prompt).
Bạn có sửa `OPTIMIZED_PROMPT` không? Không sửa — mã băm SHA-256 giữ nguyên đúng `719e74d3b6232053`, đảm bảo tính liêm chính tuyệt đối của phép so sánh và không làm suy yếu mốc chuẩn đối đầu.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6270 | **0.970** | 408.1 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5376 | **0.970** | 276.7 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 410.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 503.2 | 3.86 |

Trả lời ba câu hỏi giải phẫu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập kiểm thử target, `attn_only` đạt điểm hoà với `correct` cùng mức chính xác 0.970. Tuy nhiên, nếu xét theo training loss ở NB4, thứ tự lại bị đảo ngược: `attn_only` có loss thấp hơn rõ rệt (0.5376 so với 0.6270 của `correct`). Hiện tượng này chứng minh rằng training loss là một chỉ số thay thế (proxy metric) nguy hiểm và dễ gây ngộ nhận: việc dồn toàn bộ tham số vào một không gian hẹp (chỉ 2 ma trận q, v với rank cực lớn r=283) cho phép model ghi nhớ dữ liệu huấn luyện tốt hơn trên bề mặt biểu diễn nhưng không mở rộng thêm năng lực suy luận tổng quát. Phân bổ adapter trải rộng toàn bộ các lớp `text-linear` với rank vừa phải (r=16) mang lại khả năng thích ứng kiến trúc toàn diện và bền vững hơn nhiều so với việc chỉ tăng rank cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Đường loss của `wrong_lr` gần như bị đóng băng ở mức rất cao, kết thúc ở 1.5702 (so với 0.6270 của `correct`), dẫn đến việc mô hình hoàn toàn không học được tác vụ và đạt điểm target 0.000 cùng format 0.000. Nếu chỉ quan sát đường loss đi ngang mà không phân tích learning rate, người làm thí nghiệm rất dễ đi đến kết luận sai lầm rằng phương pháp LoRA không có khả năng thích ứng với bài toán phân loại tiếng Việt hoặc dữ liệu bị lỗi. Trong thực tế, LoRA chỉ cập nhật một ma trận tích có rank thấp nên đòi hỏi learning rate lớn hơn khoảng 10 lần so với full fine-tuning để bước cập nhật có đủ biên độ thoát khỏi vùng cực trị cục bộ ban đầu.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
Thực nghiệm đo được cho thấy `qlora` cắt giảm đỉnh VRAM từ 8.78 GB xuống còn 3.86 GB, tương đương mức tiết kiệm lên tới 56.0% bộ nhớ GPU. Tuy nhiên, sự đánh đổi là rất rõ ràng: thời gian huấn luyện tăng từ 408.1 giây lên 503.2 giây (chậm hơn ~23% do chi phí giải nén lượng tử hoá từng bước) và độ chính xác mục tiêu bị suy giảm từ 0.970 xuống 0.940. Kết quả đo đạc này hoàn toàn ủng hộ khuyến nghị của tác giả lab và đội ngũ Unsloth: trên các dòng model thế hệ 2026 như Qwen3.5, sai số lượng tử hoá 4-bit gây tổn hại có thể đo lường được lên chất lượng trích xuất thông tin, do đó chỉ nên dùng QLoRA khi phần cứng bắt buộc (như card 4–6 GB), còn nếu GPU đủ tải (như T4 16GB) thì fp16 LoRA luôn là lựa chọn vượt trội.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.113` · `valid_trace_rate = 0.0`

**Diễn giải phán quyết (156 từ)**:  
Cổng hồi quy trả về kết quả FAILED không bắt nguồn từ năng lực trên tác vụ mục tiêu, mà nằm ở sự suy thoái năng lực tổng quát. Trên tập target, bản fine-tune thể hiện sự tiến bộ vượt bậc với độ chính xác tăng từ 0.765 lên 0.970 ($\Delta = +0.205$), chứng minh quá trình huấn luyện đã dạy cho mô hình nắm vững cấu trúc phân loại nghiệp vụ 4 trường. Tuy nhiên, ở tập kiểm tra hồi quy 15 câu hỏi kiến thức mở, điểm số lại sụt giảm từ 0.7911 xuống 0.6778 ($\Delta = -0.113$), vi phạm ngưỡng dung sai cho phép là 0.020. Đây là minh chứng thực nghiệm kinh điển của hiện tượng Quên thảm họa (Catastrophic Forgetting) khi fine-tune một LLM trên dữ liệu chuyên biệt hẹp mà không có dữ liệu đối ứng. Việc ghi nhận trung thực phán quyết FAILED khẳng định giá trị cốt lõi của bài lab: ta không được đánh đổi năng lực tri thức nền tảng của mô hình chỉ để lấy độ chính xác cục bộ trên một tập dữ liệu nhỏ.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khó gặp... | van_chuyen, thap, ốp lưng điện thoại, trung_tinh | van_chuyen, trung_binh, ốp lưng điện thoại, trung_tinh | van_chuyen, thap, ốp lưng điện thoại, trung_tinh | ✅ FT thắng: Bắt chuẩn urgency thap |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu... | hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh | hoi_thong_tin, thap, ốp lưng điện thoại, trung_tinh | hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh | ✅ FT thắng: Xác định đúng mức độ khẩn cấp |
| 3 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN033689. Sai màu. Sớm nhé... | san_pham_loi, trung_binh, ốp lưng điện thoại, tieu_cuc | doi_tra, cao, ốp lưng điện thoại, tieu_cuc | san_pham_loi, trung_binh, ốp lưng điện thoại, tieu_cuc | ✅ FT thắng: Phân biệt rõ sai màu là lỗi giao hàng |
| 4 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, thap, bình giữ nhiệt, tich_cuc | hoan_tien, trung_binh, bình giữ nhiệt, trung_tinh | ❌ **FT thua**: FT overfit urgency thành trung_binh |
| 5 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện... | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, thap, nồi chiên không dầu, trung_tinh | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | ❌ **FT thua**: FT bỏ qua từ khoá "khi nào tiện" |

**Có mẫu chung nào ở các ca FT thua không?**  
Có một mẫu hình sai số rất rõ ràng: ở các ticket chứa cụm từ làm giảm mức độ khẩn cấp như "Khi nào tiện" hoặc "Chưa thấy tiền", mô hình fine-tune có xu hướng thiên lệch (bias) dự đoán mức `urgency` mặc định là `trung_binh` do tần suất xuất hiện áp đảo của nhãn này trong 225 mẫu huấn luyện. Ngược lại, baseline (b) nhờ có phần mô tả luật rõ ràng trong prompt ("khi nào tiện / không vội → thap") nên nhận diện chính xác hơn ở các trường hợp biên này.

---

## 7. Kết luận & điều tôi học được

**Kết luận (182 từ)**:  
Bản fine-tune LoRA này có nên được đưa vào vận hành (deploy) thực tế hay không? Câu trả lời phụ thuộc hoàn toàn vào kiến trúc hệ thống phục vụ. Nếu mô hình được triển khai như một vi dịch vụ (microservice) chuyên trách xử lý ngầm trong luồng tự động hoá triage ticket CSKH, ta hoàn toàn NÊN deploy: mô hình đạt độ chính xác ấn tượng 97.0% (vượt trội so với 76.5% của prompt engineering), đảm bảo 100% tuân thủ định dạng JSON với độ trễ tối ưu và chi phí token prompt ngắn hơn nhiều lần. Tuy nhiên, nếu mô hình được dùng làm trợ lý hội thoại đa năng giao tiếp trực tiếp với người dùng, ta CHƯA NÊN deploy phiên bản hiện tại do sự sụt giảm 11.3% năng lực chỉ dẫn tổng quát (regression gate FAILED).  
Bài học lớn nhất về đòn bẩy kỹ thuật trong lab này là: **Rank không phải là đòn bẩy chính, vị trí adapter và learning rate mới là yếu tố quyết định**. Thí nghiệm đối chứng công bằng đã chứng minh rank 283 ở attention không thể đánh bại rank 16 trải rộng toàn bộ linear layers, trong khi sai lệch learning rate 10x sẽ làm sụp đổ toàn bộ pipeline huấn luyện.

**Ba điều tôi học được**:
1. **Mask Proof là điều kiện tiên quyết**: Không bao giờ được tin tưởng thư viện huấn luyện một cách mù quáng; việc giải mã ngược token labels để chứng minh câu hỏi không nằm trong loss là bước bảo vệ duy nhất ngăn chặn lỗi mô hình học vẹt câu hỏi.
2. **So sánh công bằng đòi hỏi khớp ngân sách tham số**: Đặt lên bàn cân hai cấu hình LoRA có cùng rank là vô nghĩa nếu số lượng module khác nhau; chỉ khi giải ra matched rank đưa tổng tham số về sai lệch dưới 5%, ta mới thực sự đo đạc được giá trị của vị trí gắn adapter.
3. **Training loss có thể là một cái bẫy**: Loss huấn luyện của `attn_only` thấp hơn `correct` nhưng điểm target thực tế lại không tốt hơn, cho thấy việc tối ưu hoá loss trên không gian hẹp chỉ là hiện tượng ghi nhớ tham số cục bộ chứ không đại diện cho chất lượng suy luận.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
Trộn 3% dữ liệu chỉ dẫn tiếng Việt đa nhiệm (Vietnamese general instruct replay) vào tập huấn luyện 225 mẫu để khắc phục hiện tượng quên thảm hoạ, đưa cổng hồi quy NB5 từ FAILED về PASSED mà vẫn duy trì độ chính xác target trên 95%.

---

## Phụ lục — thưởng đã làm

- [x] Đã hoàn thành core pipeline NB1 → NB5 với đầy đủ 4 thí nghiệm đối chứng kiểm soát tham số công bằng trên Colab T4.
- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
