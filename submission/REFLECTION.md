# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _NGUYEN NGOC HAN_
**Khoá:** _A20-K4 / 2A202602511_
**Tier đã chạy:** _T4_
**Ngày:** _2026-10-08_

> Số liệu ghi dưới đây được trích từ output notebook `Lab22_DPO_T4.ipynb`. Những mục không có output đi kèm được đánh dấu chưa có, không suy diễn.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4; runtime cho biết 14.563 GB tổng khả dụng, không ghi VRAM đỉnh tiến trình |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1,000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9%; median 94 so với 86 token |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | Không có kết quả NB4 trong output đã lưu. Notebook cấu hình mặc định panel Skywork Reward V2 Qwen3-4B + Llama-3.2-3B |
| Chi phí | Colab Free |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | Không được ghi trong output đã lưu |
| VRAM cao nhất | Không có số đỉnh sử dụng; runtime báo T4 với 14.563 GB tổng khả dụng |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.0957 (chosen 0.3814 − rejected 0.2857) |
| Độ chính xác reward trên held-out | 0.700 |
| Margin trên held-out | 0.0824 (chosen 0.3933 − rejected 0.3109) |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | Không có output NB4 được lưu |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Các reward train bắt đầu gần 0 vì policy khởi đầu trùng với mô hình tham chiếu SFT. Ở cuối quá trình, `rewards/chosen` đạt +0.3814 và `rewards/rejected` +0.2857; gap train là +0.0957. Trên held-out, chosen đạt +0.3933 và rejected đạt +0.3109, cho margin +0.0824 cùng reward accuracy 0.700. Như vậy cả hai reward đều dương và held-out đi cùng xu hướng train; margin không hình thành vì chosen reward giảm trong khi rejected giảm nhanh hơn. Chẩn đoán `INTENDED` phù hợp với các giá trị cuối, không phải `LIKELIHOOD DISPLACEMENT` hay `FAILURE`. Margin held-out thấp hơn train một chút, nhưng các số cuối không cho thấy chỉ train mới học được preference. Ảnh đường cong không có trong snapshot repo, nên nhận xét này dựa trên metrics cuối notebook in ra, không khẳng định hình dạng đường ở từng bước. Accuracy 70% cũng cho thấy tín hiệu preference trên held-out chưa hoàn hảo.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | Chưa có kết quả NB4 | — | — | — | — | — | — |
| hữu ích — helpfulness (4) | Chưa có kết quả NB4 | — | — | — | — | — | — |
| an toàn — safety (4) | Chưa có kết quả NB4 | — | — | — | — | — | — |

Giám khảo/sanity accuracy: không có kết quả NB4 được lưu. Notebook mặc định cấu hình panel Skywork Reward V2 Qwen3-4B và Llama-3.2-3B; chưa xác nhận panel đã chạy.


---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | — | — | — | Sweep chưa chạy |
| 0.1 | 0.0824 | 0.700 | INTENDED | Run DPO chuẩn |
| 0.5 | — | — | — | Sweep chưa chạy |

Chưa chạy sweep. Tôi dự đoán β=0.05 sẽ tạo cập nhật nhẹ hơn, có thể giữ sát SFT nhưng margin nhỏ hơn. β=0.5 có thể làm margin tăng nhanh hơn nhưng cũng tăng nguy cơ thay đổi hành vi/độ dài mạnh hoặc giảm chất lượng tổng quát. β=0.1 có khả năng ở giữa hai trường hợp; cần so sánh held-out accuracy và đầu ra sinh chứ không chỉ chọn margin lớn nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của tôi là dùng β=0.1 cho DPO. Phương án khác là β thấp hơn (0.05), tạo cập nhật chính sách nhẹ hơn, hoặc β cao hơn (0.5), tăng áp lực phân biệt chosen và rejected so với reference. Tôi chọn 0.1 làm điểm cân bằng khi huấn luyện một epoch trên 800 cặp với LoRA, thay vì bắt đầu bằng cập nhật quá mạnh. Kết quả cho thấy thiết lập này học được tín hiệu preference: loss giảm từ 0.6935 xuống 0.6756, reward gap train kết thúc ở 0.0957, và held-out margin vẫn dương 0.0824 với reward accuracy 0.70. Cả train và held-out đều có chosen/rejected reward dương, nên margin dương không cần đẩy cả hai log-ratio xuống dưới reference. Tuy vậy, không thể kết luận β=0.1 tốt hơn các giá trị thay thế vì thiếu beta-sweep; cũng chưa chứng minh được chất lượng câu trả lời cho người dùng cải thiện vì thiếu NB4. Nếu làm lại, tôi sẽ chạy sweep với split, seed và lượng huấn luyện cố định; so sánh margin, accuracy held-out và độ dài đầu ra, rồi chọn β không chỉ theo margin cao nhất mà còn theo đánh giá độc lập và kiểm tra độ dài. Như vậy quyết định dựa trên chất lượng tổng thể thay vì tối ưu riêng một chỉ số loss.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | Chưa có kết quả | — | — | — |
| GSM8K | Chưa có kết quả | — | — | — |
| Global-MMLU-vi | Chưa có kết quả | — | — | — |

Không có output NB6 hay `benchmark_results.json`, nên tôi không thể báo điểm SFT/SFT+DPO, stderr hoặc Δ. Vì vậy chưa thể xác định chênh lệch nào vượt khoảng 2×stderr hoặc liệu có alignment tax trên GSM8K hay không. Điểm benchmark có nhiễu lấy mẫu; chênh lệch nhỏ hơn sai số không đủ để kết luận năng lực tăng hoặc giảm. Benchmark cũng đo năng lực khác với reward accuracy của preference dataset, nên không thể thay bằng margin DPO 0.0824 hay accuracy 0.70. Kết quả này chưa được chạy/lưu, không phải điểm 0. Nếu hoàn thành NB6, tôi sẽ ghi giới hạn đánh giá, điểm và stderr của hai model, đồng thời kiểm tra chat template theo rubric. Sau đó sẽ xét Δ theo đúng dấu và độ lớn tương đối với stderr rồi đối chiếu xu hướng NB4. Với bằng chứng hiện có, chưa có cơ sở kết luận alignment tax hoặc cải thiện benchmark.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 0.69 | 0.0225 | 442.3 chars | INTENDED |
| RPO | 0.66 | 0.0362 | 433.5 chars | INTENDED |
| DPO-norm | 0.64 | 0.0098 | 438.55 chars | LIKELIHOOD DISPLACEMENT |
| LD-DPO | 0.56 | 0.0263 | 441.95 chars | LIKELIHOOD DISPLACEMENT |
| ORPO | Không có output lưu | — | — | Chưa xác minh kết quả |

Trong các biến thể có số liệu, độ dài trung bình nằm trong khoảng 433.5–442.3 ký tự: RPO ngắn nhất và DPO dài nhất. Không có độ dài SFT trên cùng prompt nên không thể tính mức thay đổi thực so với SFT. RPO thêm NLL trên chosen bên cạnh loss DPO để khuyến khích giữ xác suất câu chosen, có thể hạn chế likelihood displacement. DPO-norm chuẩn hoá theo độ dài token và LD-DPO điều chỉnh trọng số token, nhưng hai run vẫn được chẩn đoán likelihood displacement. Do thiếu ORPO và baseline SFT, không thể xác định biến thể thay đổi độ dài nhiều nhất một cách đầy đủ.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | Chưa có output GRPO lưu |
| Sai số chuẩn ≈ √(p(1−p)/n) | Chưa tính được khi thiếu n và p |

Không có log reward hoặc accuracy trước/sau GRPO trong output đã lưu, nên chưa thể xác định reward định dạng hay reward đáp án tăng trước, hoặc chênh lệch có vượt nhiễu hay không.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8; có DPO/RPO/DPO-norm/LD-DPO metrics)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Reward margin held-out dương và diagnosis là `INTENDED`, nhưng held-out reward accuracy chỉ đạt 0.70. Điều này nhắc tôi rằng margin tăng không đồng nghĩa mọi cặp preference được xếp đúng, càng không tự động chứng minh người dùng sẽ thích câu trả lời DPO hơn.
