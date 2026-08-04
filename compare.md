Xác minh tính xác thực của thông tin kiến trúc hệ thống được công bố
Xác minh tính xác thực của thông tin kiến trúc hệ thống được công bố

Trước khi so sánh, mình cần verify lại thông tin này — lần trước mình search kỹ mà không thấy paper hay báo cáo kiến trúc kỹ thuật nào của OpenCubee_1, nên để chắc chắn không so sánh dựa trên thông tin sai, mình sẽ search lại cụ thể theo tên các model bạn vừa nêu.

Phân tích kiến trúc OpenCubee, so sánh thành phần, đề xuất tối ưu hóa
Phân tích kiến trúc OpenCubee, so sánh thành phần, đề xuất tối ưu hóa
Đã verify — thông tin bạn đưa ra là chính xác

Đội OpenCubee_1 đã lựa chọn hướng tiếp cận xây dựng một hệ thống tích hợp Conversational Intelligence Agent (ConvAgent) hoạt động song song với các phương thức tìm kiếm truyền thống, đóng vai trò như một trợ lý thông minh giúp tự động hóa hoàn toàn quy trình tìm kiếm. Thay vì người dùng phải tự lựa chọn công cụ thủ công, Agent sẽ phân tích ý định để kích hoạt các bộ lọc phù hợp nhất từ ngữ nghĩa, OCR, ASR, cho đến phát hiện vật thể — dùng BEiT-3, MetaCLIP-2, Perception Encoder cho ngữ nghĩa; Chunkformer cho ASR; DeepSolo + PARSeq cho OCR. 
UIT
UIT

Một lưu ý quan trọng cần nói thẳng trước khi so sánh: nguồn này là bài phỏng vấn/PR sinh viên của phòng tuyển sinh UIT, không phải paper kỹ thuật. Nó cho biết dùng model gì nhưng hoàn toàn không tiết lộ: cách fusion/rank giữa 3 model embedding, thuật toán clipping, database dùng gì, cách xử lý time-budget, hay có cơ chế feedback loop không. Nên phần so sánh dưới đây sẽ trung thực về chỗ nào biết chắc, chỗ nào là khoảng trống không công khai.

Bảng so sánh trực diện
Thành phần	OpenCubee_1 (vô địch AIC 2025)	Kiến trúc bạn đang build (BT2/AIC 2026)	Nhận xét
Intent/Routing	1 agent duy nhất (ConvAgent) vừa phân tích ý định vừa tự kích hoạt nhánh	Router Agent + Planner Agent tách riêng (theo doc trước)	OpenCubee gộp 2 vai trò làm 1 — đơn giản hơn, có bằng chứng thực chiến đã thắng. Nên cân nhắc gộp lại để giảm latency, chỉ tách logic bên trong code chứ không cần 2 agent riêng về kiến trúc
Visual embedding	Ensemble 3 model: BEiT-3 + MetaCLIP-2 + Perception Encoder	1 model: SigLIP	Ensemble tăng độ phủ/robust với query lạ (out-of-knowledge), khớp với phát hiện của Mixpeek Benchmark là 1 embedding đơn lẻ dễ yếu — nhưng tốn compute+storage gấp 3
ASR	Chunkformer	WhisperX	Chunkformer hiệu quả hơn cho video dài (324 giờ dữ liệu AIC), nhưng cần kiểm tra độ chính xác word-level timestamp — quan trọng cho phần đồng bộ audio↔shot đã bàn ở trên
OCR	DeepSolo (detect) + PARSeq (recognize), tự fine-tune PARSeq cho tiếng Việt	Gemini OCR (VLM 1-shot)	Fine-tune riêng kiểm soát tốt hơn, không phụ thuộc API, không tốn cost/rate-limit mỗi frame — nhưng cần công train + GPU riêng (họ được UIT cấp GPU)
Fusion/Clipping algorithm	❌ Không công bố	RRF/CC + Unified Clipping (từ U-CESE, đã verify chi tiết)	Đây là chỗ bạn có lợi thế rõ ràng — bạn có tài liệu học thuật cụ thể để implement, OpenCubee là black-box
Multi-agent debate/verification	Không có — chỉ 1 ConvAgent, không multi-agent debate kiểu MAVIS	Đề xuất Verification Agent riêng	Đội vô địch thực tế không cần multi-agent debate phức tạp — củng cố khuyến nghị trước đó của mình: giữ verification nhẹ, ưu tiên tốc độ
Automated/no-human-loop mode	Không cần (2025 chưa có hình thức tự động)	Cần Answer/Submission Agent (bắt buộc cho 2026)	Khoảng trống lớn nhất — không có gì để học từ OpenCubee ở đây vì họ chưa từng phải giải bài toán này
UI/UX	Được đội vô địch nhấn mạnh là "chìa khóa chiến thắng" — "hãy đầu tư vào Giao diện, một hệ thống Backend mạnh đến đâu cũng cần một Frontend nhanh và dễ dùng để phát huy tác dụng" 
UIT
	Chưa đề cập trong doc gửi Huy Phạm	Cần bổ sung — bài học thực chiến trực tiếp, không phải suy đoán
Ưu điểm của kiến trúc OpenCubee_1
Đơn giản hóa tối đa số lượng agent (1 thay vì nhiều) → ít điểm fail, ít latency, dễ debug khi thi đấu dưới áp lực thời gian thực (chính họ kể đã gặp sự cố kỹ thuật ở vòng chung kết và phải gỡ gạc).
Ensemble 3 backbone visual khác nhau (BEiT-3/MetaCLIP-2/Perception Encoder) thay vì đặt cược vào 1 model — đúng tinh thần lời khuyên của họ: "đừng phụ thuộc vào một model duy nhất".
Fine-tune OCR riêng cho tiếng Việt thay vì dựa hoàn toàn vào VLM ngoài — giảm rủi ro phụ thuộc API và noise domain-specific.
Nhược điểm / rủi ro nếu áp dụng y nguyên vào 2026
Toàn bộ kiến trúc này được tối ưu cho hình thức truyền thống (người vận hành) — chưa hề được test trong kịch bản tự động đối đầu trực tiếp. Nếu chỉ copy nguyên xi, bạn vẫn thiếu chính xác thứ AIC 2026 mới thêm vào.
Ensemble 3 model embedding sẽ tăng đáng kể thời gian indexing (324 giờ video × 3 lần forward pass) — cần cân nhắc compute budget trước khi copy nguyên.
Không công khai cách fusion/rank giữa 3 embedding — nếu bạn muốn học theo hướng ensemble, phải tự thiết kế phần này (có thể dùng RRF bạn đã có sẵn, coi 3 model như 3 "kênh" độc lập giống cách Planner đang tách kênh OBJ/SCN/ACT).
Có chắp vá được không? — Có, cụ thể như sau
Gộp Router + Planner thành 1 agent kiểu ConvAgent — giữ logic decomposition sub-intent ở tầng code bên trong, nhưng expose ra ngoài chỉ 1 agent quyết định kích hoạt nhánh nào. Giảm 1 lượt gọi LLM, giảm latency.
Thêm 1 model embedding thứ 2 bên cạnh SigLIP (ví dụ Perception Encoder — mạnh nhất trong 3 model họ dùng theo benchmark chung của cộng đồng) và fuse 2 kênh này bằng RRF sẵn có — không cần 3 model ngay, chỉ cần 2 đã tăng robustness đáng kể mà chi phí thêm không quá lớn.
Giữ nguyên Unified Clipping + RRF/CC — đây là phần bạn hơn hẳn OpenCubee vì có literature cụ thể để dựa vào.
Answer/Submission Agent — bắt buộc tự thiết kế, không có gì để chắp vá từ OpenCubee vì họ chưa từng cần.
Bổ sung ưu tiên UI/UX tốc độ cao vào roadmap — đây là input mới, nên thêm vào mục "Next step" trong file đã gửi Huy Phạm.
Cân nhắc fine-tune riêng 1 module (ví dụ OCR hoặc ASR tiếng Việt) thay vì phụ thuộc hoàn toàn Gemini API — nhưng làm sau khi baseline chạy ổn, vì đây là khoản đầu tư công sức lớn (cần GPU, dataset, thời gian train) đúng như OpenCubee đã có lợi thế được trường cấp GPU.

Nếu muốn, mình cập nhật luôn phần này vào file đã gửi Huy Phạm (thêm mục "Bài học từ đội vô địch OpenCubee_1" + cập nhật roadmap UI/UX) để giữ mọi thứ tập trung ở một chỗ.