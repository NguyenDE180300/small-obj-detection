# LEVIR-Ship: YOLOv8-P2 baseline vs Adaptive-SRM variants

The Jetson Orin FastAPI inference service and workstation deployment script are
documented in [`inference_bundle/jetson_api/README.md`](inference_bundle/jetson_api/README.md).

Nền thực nghiệm dùng YOLOv8n-P2 với YAML `yolov8n_p2_levir_baseline.yaml`,
`Detect` chuẩn và pretrained transfer qua `model.load(..., smart_transfer=True)`:
3,352,396 tham số và 358 tensor transfer. Toàn bộ ASRM case chỉ thêm module/layer
routing trên cùng nền này.

> **Naming policy:** YAML và tên experiment công khai dùng tiền tố `baseline_` hoặc
> tên module/mục tiêu kỹ thuật; không dùng tên cá nhân hoặc tên repository bên ngoài.

Code training độc lập cho phía YOLO, chạy song song với `Levir_ship_training/` (phía
mmdet). Dùng bản Ultralytics local đóng gói sẵn trong `ultralytics/` (dựa trên
`YOLO_custom_org`, vanilla v8.4.67 + các module ASRM ở `model_cfg/README.md`), không
`pip install ultralytics` để tránh đè lên bản đã có sẵn trong môi trường đích.

> **Augmentation policy (từ 08/09/2026):** mọi run mới qua `run_experiment.py`
> đều dùng `mosaic=0.0`. Các run lịch sử vẫn giữ nguyên augmentation đã dùng khi
> train, vì vậy không được gộp trực tiếp với nhóm no-mosaic nếu mục tiêu là ablation
> augmentation công bằng. Case `baseline_gap_factorized_k15_nomosaic` giữ như một tên
> explicit để truy vết baseline no-mosaic đầu tiên.

Ma trận 10 case × 3 seed (42/43/44), cùng data/epoch/batch/image-size/pretrained/scale
('n') để so sánh công bằng:

| Case | Kiến trúc |
|---|---|
| `baseline_p2` | YOLOv8-P2 gốc (`ultralytics/cfg/models/v8/yolov8-p2.yaml`), không đổi |
| `p2_asrm_fusion` | Version 1: SRM residual gated vào **cả P2 và P3** (2 tap độc lập) |
| `input_asrm_auxiliary` | Version 2: SRM trên ảnh gốc, attention phụ trợ, chỉ P2, không đổi detection path |
| `input_asrm_guided_p2` | Version 3: SRM trên ảnh gốc, attention guide **cả P2 và P3** (train + eval) |
| `asrm_candidate_sparse_refine_p2p3` | K4/K5/K8 candidates → P2/P3 detail + P4/P5 background prototypes → verifier → sparse refine P2/P3 |

Chi tiết kiến trúc, tham số constructor, cách các module ảnh+level lấy ảnh gốc mà không
cần wrapper Model: xem `model_cfg/README.md`.

## Chạy chung với Varroa trong 1 notebook

`../ASRM_matrix_both_datasets.ipynb` (repo root) orchestrate cả project này và
`../Varroa_asrm_training/` trong 1 lần chạy: tải code+data riêng của từng dataset từ
đúng HF repo, prepare bằng script riêng, train 4 case × seed tuần tự cho từng dataset
(giữ nguyên hyperparameter/split đã verify riêng), rồi `../combine_results.py` gộp 2
CSV aggregate (gắn cột `dataset`) thành 1 DataFrame duy nhất — không gộp trị số mAP
giữa 2 dataset, chỉ đặt cạnh nhau.

## Setup baseline thống nhất

`run_experiment.py` dùng `pretrained=yolov8n.pt`, `imgsz=512`, `batch=8`,
`epochs=100`, `patience=20` và ba seed 42/43/44. Các case so sánh cùng
transfer-learn, input size, batch, seed và **fixed data split**; khác biệt chủ đích
là kiến trúc/loss của từng experiment.

`prepare_yolo_data.py` sort 3.896 crop stem, dùng `random.Random(42).shuffle(stems)`,
rồi cắt tuần tự train/val/test = **2320/788/788**. Split seed 42 là cố định và dùng
chung cho cả ba training seed 42/43/44. Đây là random split theo crop nên vẫn có nguy
cơ crop từ cùng scene xuất hiện ở nhiều split.

### Pretrained transfer của từng case

| Case | Transfer từ `yolov8n.pt` |
|---|---|
| `baseline_p2` | load trực tiếp theo key + shape; backbone giữ index `0..9` |
| `p2_asrm_fusion` | load trực tiếp; module ASRM nằm sau backbone nên không dịch index |
| `input_asrm_auxiliary` | remap backbone `model.i.* → model.(i+1).*` vì có `Identity@0` |
| `input_asrm_guided_p2` | remap backbone `model.i.* → model.(i+1).*` vì có `Identity@0` |
| `asrm_candidate_sparse_refine_p2p3` | remap backbone `model.i.* → model.(i+1).*` vì có `Identity@0` |

Remap chỉ load tensor có key đích và shape trùng khớp, đồng thời báo lỗi nếu không
transfer được tensor nào. Các layer P2/head/ASRM không tồn tại tương ứng trong
`yolov8n.pt` vẫn dùng initialization riêng; tham số `--pretrained ''` mới là chế độ
train from scratch.

Kết luận benchmark nên dựa trên mean±std của ba seed trong
`comparison_yolo_aggregate.csv`. `delta_bbox_mAP_mean` luôn lấy `baseline_p2` cùng
fixed split làm mốc.

## Luồng chạy (trong `Levir_ship_training_YOLO.ipynb`)

1. Tải `code/Levir_ship_training_YOLO_code.zip` (chứa thư mục này) và
   **`data/levir_ship_data.zip` đã có sẵn trên HF** — dùng lại đúng file data mmdet đang
   dùng, không tải/nhân bản dữ liệu ảnh lần hai.
2. `pip install -r requirements.txt`.
3. `prepare_yolo_data.py` build layout `images/{train,val,test}` + `labels/{train,val,test}`
   + `data.yaml` bằng **symlink** (không copy 863MB ảnh) từ `LevirShipData/All Images` +
   `All Annotations`, dùng fixed random crop split seed 42; file
   `split_manifest.json` lưu toàn bộ stem để kiểm tra.
4. `run_experiment.py --case <case> --seed <seed> --data-yaml data.yaml --pretrained
   yolov8n.pt ...` transfer-learn + train + test 1 (case, seed) qua
   `ultralytics.YOLO(...).train()/.val(split="test")`, ghi `test_metrics.json` vào
   `work_root/<case>/seed_<seed>/`.
5. `collect_results.py --seeds 42 43 44` gom 12 run thành CSV thô + CSV tổng hợp
   mean±std (`*_aggregate.csv`), có `delta_bbox_mAP_mean` so với `baseline_p2`.

## Trạng thái đã kiểm chứng (trước khi đưa lên HF)

- `prepare_yolo_data.py` đã chạy trên bản data mirror local, ra đúng 2320/788/788 và
  không trùng stem giữa split.
- Cả 4 config đã chạy **thật** qua `ultralytics.YOLO(...).train()` + `.val(split="test")`
  (CPU, 1 epoch, ảnh thật cỡ nhỏ) — dataloader đọc đúng label, loss/backward/checkpoint/
  validate không lỗi cho cả baseline lẫn 3 biến thể ASRM.
- Pretrained-weight transfer (`load_pretrained` trong `run_experiment.py`) đã test
  riêng: so sánh trực tiếp giá trị tensor sau khi load với checkpoint `yolov8n.pt` gốc,
  đúng cho cả 4 case — bao gồm 2 case cần remap index (`input_asrm_auxiliary`,
  `input_asrm_guided_p2`, xem `model_cfg/README.md`).
- `collect_results.py` đã test với dữ liệu tổng hợp 4 case × 3 seed, ra đúng CSV thô +
  aggregate mean/std.
- **Chưa train full-scale trên GPU** — đây là bước chuẩn bị pipeline, chưa phải kết quả
  benchmark.
- Input ASRM Auxiliary (Version 2) mới chỉ expose `last_attention_map` trên module —
  chưa có auxiliary loss thật, nên trong ma trận này nó sẽ train giống hệt baseline (P2
  không đổi); phần khác biệt tiềm năng nằm ở việc bạn tự thêm loss dùng attention map đó
  sau này.

## Case mới: RGB + Saturation + Quality

`baseline_gap_factorized_k15_rgb_saturation_quality` giữ nguyên RGB+Saturation (4-channel
first Conv), FPN-only P5→P2, GAP ChannelAttention, P2-only Detect và Factorized TAL
K15. Khác biệt duy nhất là bật quality head IoU có sẵn: `quality_head=true` và
`quality_score_mode=cls_mul_q`, nên score suy luận là
`sigmoid(class_logit) * sigmoid(quality_logit)`. Quality target dùng IoU của box dự
đoán với GT cho positive assignment và 0 cho background; dùng BCE mặc định với
`quality_gain=0.5`, không bật hard-negative mining. Input path, backbone, neck,
augmentation và pretrained remap không thay đổi.

## Ba case saturation-guidance mới

- `baseline_gap_factorized_k15_rgb_saturation_stem`: backbone nhận RGB 3 kênh; saturation
  đi qua stem CNN nhỏ riêng và được cộng residual vào backbone P2 với `gamma=0`.
- `baseline_gap_factorized_k15_rgb_saturation_guided_p3_residual`: saturation chỉ tạo gate
  không gian; nội dung residual đến từ semantic F3 và được cộng vào F2 cuối với `gamma=0`.
- `baseline_gap_factorized_k15_rgb_saturation_feature_filter`: selector softmax chọn trộn
  F3/F4 theo từng vị trí, sau đó mask sigmoid lọc context trước khi cộng residual vào F2.

Cả ba giữ P5/SPPF, FPN-only P5→P2, GAP, Detect P2-only stride 4 và FTAL K15; không
thêm PAN, ASRM, quality head hay detection scale mới. Các stem/filter đều nằm trên
đường train và inference, nhưng bắt đầu gần identity nhờ gain học được khởi tạo 0.

### P2→P3 guidance ablations

- `baseline_gap_factorized_k15_rgb_saturation_filtered_guided_p3_residual`: P2 gốc vẫn
  đi thẳng đến fusion P2 cuối; một bản P2 đã lọc chỉ tồn tại trong guidance branch,
  được downsample và gated trước khi residual-add vào P3 với `alpha3=0`.
- `baseline_gap_factorized_k15_rgb_saturation_dual_gate_p3_residual`: không tạo filtered
  P2 replacement. Mask `Mf` chọn P2 đáng tin, mask `Mg` kiểm tra tương thích P2/P3;
  `Mf_down * Mg * P2_down` là residual được thêm vào P3 với `alpha3=0`.

### Post-FPN F2 saturation ablations

Ba case sau không sửa P2 backbone. Chúng lấy `F2_base` sau normal FPN P5→P4→P3→P2,
sau đó residual-add trước GAP/Detect: `stem_f2` thêm trực tiếp saturation stem;
`guided_p3_f2` dùng F3 làm content với gate từ F2+saturation; `regularized_f2_fusion`
dùng C_mid nhỏ, GN, depthwise selector và Dropout2d để giới hạn overfit. Mọi gain bắt
đầu bằng 0 nên `F2_out == F2_base` chính xác tại khởi tạo.
