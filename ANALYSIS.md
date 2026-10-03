# Báo cáo Đánh giá & Phân tích Hệ thống Memory cho AI Agent (Day 17)

**Học viên:** Nguyễn Đức Phát  
**MSSV:** 2A202602753  
**Lớp:** K4-L3A  
**Track:** Phase 2, Track 3 - Day 17: Memory Systems for AI Agent  

---

## 1. Kiến trúc Hệ thống Memory 3 Tầng (Tri-Layer Memory Architecture)

Hệ thống được thiết kế để giải quyết bài toán trade-off giữa **độ nhớ dài hạn (cross-session recall)**, **chất lượng phản hồi (quality)**, và **chi phí token (prompt token explosion)**.

```
                    +------------------------------------+
                    |        Incoming User Message       |
                    +-----------------+------------------+
                                      |
                     [Extract Facts & Noise Filter]
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
  +----------------------+                           +--------------------+
  |  UserProfileStore    |                           | CompactMemory      |
  |  (state/profiles/    |                           | Manager            |
  |   <user>/User.md)    |                           +---------+----------+
  +----------+-----------+                                     |
             | (Persistent Across Threads)                     v
             |                                       [Token Count > 120?]
             |                                        /                \
             |                                     YES                  NO
             |                                      /                    \
             |                           [Summarize Old Turns]       [Keep Raw]
             |                           [Keep Last 4 Turns  ]           |
             |                                      \                    /
             +--------------------+------------------+------------------+
                                  |
                                  v
           [Assemble Prompt: User.md + Summary + Kept Turns + Query]
                                  |
                                  v
                        [Generate Response]
```

1. **Short-Term Memory (Bộ nhớ ngắn hạn trong phiên):**
   - Quản lý danh sách các lượt trao đổi gần nhất trong thread hiện tại.
   - Baseline Agent chỉ có tầng này; khi đổi `thread_id`, toàn bộ ngữ cảnh cũ bị xóa trắng.
2. **Persistent Memory (Bộ nhớ bền vững qua `User.md`):**
   - Lưu trữ các facts ổn định (tên, nghề nghiệp, nơi ở, phong cách, sở thích) dưới dạng Markdown chuẩn.
   - Cho phép Advanced Agent trả lời chính xác các câu hỏi nhớ lại (*recall questions*) trong các session hoàn toàn độc lập.
3. **Compact Memory (Bộ nhớ nén cho ngữ cảnh dài):**
   - Khi tổng token của tin nhắn trong thread vượt ngưỡng (`threshold_tokens = 120`), hệ thống tự động kích hoạt nén các tin nhắn cũ thành bản tóm tắt súc tích và chỉ giữ lại `keep_messages = 4` tin nhắn gần nhất.
   - Giúp kiềm chế sự bùng nổ token ngữ cảnh (`prompt tokens processed`).

---

## 2. Kết quả Thực nghiệm Benchmark Chi tiết

Chạy trực tiếp từ mã nguồn `src/benchmark.py` trên 2 bộ dữ liệu chuẩn tiếng Việt:

### Bảng 1: Standard Benchmark (`data/conversations.json` - 10 phiên hội thoại, ~100 lượt)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 2,899 | 26,246 | 4.8% | 24.4% | 0 B | 0 |
| **Advanced Agent** | 4,680 | 48,486 | **100.0%** | **100.0%** | 371 B | 154 |

### Bảng 2: Long-Context Stress Benchmark (`data/advanced_long_context.json` - 16 lượt dài dồn dập)

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Agent** | 482 | 31,086 | 0.0% | 21.0% | 0 B | 0 |
| **Advanced Agent** | 1,775 | **21,229** | **100.0%** | **100.0%** | 288 B | 28 |

---

## 3. Phân tích Bản chất Trade-off Kỹ thuật

### 3.1. Vì sao Advanced Agent tốn token hơn ở hội thoại ngắn?
- Ở các hội thoại ngắn (như trong Standard Benchmark), mỗi lượt của Advanced Agent phải mang theo phần tiền tố cấu hình hệ thống: nội dung hồ sơ `User.md` (khoảng 350-400 bytes) + tóm tắt ngữ cảnh.
- Baseline Agent không mang vác metadata nào nên chi phí prompt ban đầu rất thấp.
- **Kết luận:** Với các tác vụ chat đơn lẻ hoặc hội thoại < 3 lượt, việc nạp Persistent Memory tạo ra chi phí phụ trội (*overhead*).

### 3.2. Vì sao Compact Memory tạo ra ưu thế vượt trội ở hội thoại dài?
- Trong Long-Context Stress Benchmark (16 lượt với nội dung kỹ thuật dày đặc):
  - **Baseline Agent:** Nhồi nhét toàn bộ lịch sử thô qua từng lượt. Tổng lượng prompt token tăng theo cấp số cộng tích lũy, đạt **31,086 tokens**.
  - **Advanced Agent:** Nhờ 28 chu kỳ compaction, các lượt cũ liên tục được thu gọn thành bản tóm tắt, chỉ giữ lại 4 lượt tin nhắn mới nhất. Kết quả là chỉ tiêu tốn **21,229 prompt tokens** (tiết kiệm **9,857 tokens**, tương đương **giảm 31.7% chi phí prompt context**).
- Đồng thời, khi sang thread mới, Baseline Agent đạt **0.0% recall** (quên sạch), trong khi Advanced Agent vẫn duy trì **100.0% recall**.

### 3.3. Rủi ro về Tăng trưởng Bộ nhớ (Memory Growth)
- Tệp `User.md` tăng trưởng rất chậm và ổn định (từ 0 lên 371 Bytes sau 10 cuộc hội thoại).
- **Rủi ro thực tế:** Nếu không có cơ chế lọc nhiễu, `User.md` sẽ bị phình to (*memory bloating*) do lưu trữ cả những chi tiết vụn vặt tạm thời, dẫn tới việc chính tệp profile làm tràn context window của LLM.

---

## 4. Tính năng Bonus Triển khai (Mục tiêu 90-100 Điểm)

### 4.1. Conflict Handling (Xử lý mâu thuẫn & Đính chính thông tin)
- **Vấn đề giải quyết:** Khi người dùng đính chính (ví dụ: *chuyển nơi ở từ Đà Nẵng sang Huế*, hoặc *chuyển nghề từ backend engineer sang MLOps engineer*), các hệ thống ngây thơ thường lưu đè cả hai facts dẫn đến câu trả lời mâu thuẫn.
- **Giải pháp:** Phương thức `UserProfileStore.upsert_facts()` phân tích cấu trúc fact dạng key-value, khi phát hiện fact mới có cùng thuộc tính (`Nơi ở`, `Nghề nghiệp`), hệ thống tự động ghi đè giá trị mới nhất, xóa bỏ hoàn toàn giá trị cũ đã lỗi thời.

### 4.2. Noise Filtering & Confidence Threshold (Lọc nhiễu & Ngưỡng tin cậy)
- **Vấn đề giải quyết:** Người dùng thường có câu đùa (*"đùa với đồng nghiệp là chuyển sang làm product manager"*), hoặc nói về chuyến đi ngắn ngày (*"Hà Nội chỉ là nơi mình vừa bay ra họp 2 ngày"*), hoặc nhắc tới thú cưng (*"con corgi tên Bơ"*).
- **Giải pháp:** 
  - Tự động bỏ qua các câu hỏi truy vấn thuần túy kết thúc bằng dấu `?` để tránh suy đoán sai thành fact.
  - Sử dụng heuristic regex phát hiện ngữ cảnh đùa cợt và di chuyển tạm thời để loại bỏ `product manager` và `Hà Nội`.
  - Phân tách rành mạch giữa tên người dùng và tên thú cưng `Bơ`.

### 4.3. Đánh giá Rủi ro của Tính năng Bonus
- **Lợi ích:** Đạt độ chính xác tuyệt đối (100% Recall & 100% Quality trên cả 2 bộ benchmark), không bị lừa bởi dữ liệu bẫy.
- **Rủi ro trong môi trường production:** Các bộ lọc dựa trên quy tắc/regex có thể gặp giới hạn khi người dùng diễn đạt bằng các cấu trúc ngữ pháp bất quy tắc hoặc tiếng lóng mới. Khi mở rộng hệ thống lớn, cần kết hợp thêm LLM-as-a-Judge hoặc Structured Entity Extraction với JSON schema nghiêm ngặt.

---

## 5. Kết quả Kiểm thử Đơn vị (`pytest src/test_agents.py`)

Tất cả **5/5 test cases** vượt qua 100%:
1. `test_user_markdown_read_write_edit`: Kiểm tra toàn diện CRUD trên tệp `User.md`.
2. `test_compact_trigger`: Kiểm tra cơ chế tự động nén khi vượt ngưỡng token.
3. `test_cross_session_recall`: Kiểm tra Advanced nhớ facts qua thread mới và Baseline quên sạch.
4. `test_compact_reduces_prompt_load_on_long_thread`: Kiểm tra sự giảm tải prompt token rõ rệt của compact memory.
5. `test_bonus_conflict_handling_and_noise_filtering`: Kiểm tra tính năng bonus đính chính fact và loại bỏ nhiễu.
