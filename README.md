# TALE POSCO 합성 실행 가이드

## 1. 준비 경로

| 구분 | 경로 예시 | 내용 |
| --- | --- | --- |
| 정상 CCTV 이미지 | `/정상데이터/폴더/경로` | 합성 배경으로 사용할 정상 이미지 |
| 물체 이미지 | `objects/` | 철로 위에 합성할 물체 이미지 |
| 물체 mask | `objects/<object>_mask.png` | 물체 영역 segmentation mask |
| 철로 mask | `background/` | 채널별 배치 가능 영역 mask |
| Stable Diffusion 모델 | `stable-diffusion-2-1-base` | diffusion 실행 모델 |

## 2. 정상 이미지 경로

```text
/정상데이터/폴더/경로/
  02/
    [CH002] image_0001.jpg
    [CH002] image_0002.jpg
  04/
    [CH004] image_0001.jpg
    [CH004] image_0002.jpg
```

- 파일명 또는 폴더명에 채널 번호 포함
- 예: `[CH002]`, `CH002`, `02`는 `CH002`로 처리

## 3. 물체 이미지 / Mask 경로

```text
objects/
  object_1.png
  object_2.jpg
  object_2_mask.png
```

지원 mask 이름:

```text
object_2_mask.png
object_2_seg.png
object_2_segmentation.png
```

## 4. 철로 Mask 경로

```text
background/
  02_mask.jpg
  04_mask.jpg
  06_mask.jpeg
  08_mask.jpg
```

## 5. 생성되는 Input 구조

```text
diffusion_result/input/
  metadata.jsonl
  ch002_object_1_0000001/
    background.png
    foreground.png
    segmentation.png
    location.png
  ch002_object_2_0000002/
    background.png
    foreground.png
    segmentation.png
    location.png
```

| 파일 | 설명 |
| --- | --- |
| `background.png` | 정상 CCTV 배경 이미지 |
| `foreground.png` | 정규화된 물체 이미지 |
| `segmentation.png` | foreground 안의 물체 mask |
| `location.png` | background 위에 물체가 들어갈 위치 mask |
| `metadata.jsonl` | sample별 prompt, 원본 경로, 결과 경로, 배치 정보 |

## 6. Input만 생성

```bash
cd /path/to/Tale
CUDA_VISIBLE_DEVICES=1 python posco_rail_diffusion.py \
  --train-dir /home/poscouser/kookmin/anomaly_detection/data/posco/train \
  --object-dir objects \
  --channel-background-dir background \
  --out-dir diffusion_result/input \
  --preview-dir diffusion_result/preview \
  --result-dir diffusion_result/result \
  --normalized-object-dir diffusion_result/objects_normalized \
  --all-backgrounds-per-object \
  --sample-name-format posco \
  --flat-results \
  --require-channel-mask \
  --prompt-template "a cardboard box on a railway track at a steel mill" \
  --seed 2 \
  --placement-size-frac 0.045 0.085 \
  --overwrite
```

생성 결과:

```text
diffusion_result/input/
diffusion_result/preview/
diffusion_result/objects_normalized/
```

## 7. Input 생성 후 Diffusion까지 실행

```bash
cd /path/to/Tale
CUDA_VISIBLE_DEVICES=1 python posco_rail_diffusion.py \
  --train-dir /home/poscouser/kookmin/anomaly_detection/data/posco/train \
  --object-dir objects \
  --channel-background-dir background \
  --out-dir diffusion_result/input \
  --preview-dir diffusion_result/preview \
  --result-dir diffusion_result/result \
  --normalized-object-dir diffusion_result/objects_normalized \
  --all-backgrounds-per-object \
  --sample-name-format posco \
  --flat-results \
  --require-channel-mask \
  --prompt-template "a cardboard box on a railway track at a steel mill" \
  --seed 2 \
  --placement-size-frac 0.045 0.085 \
  --overwrite \
  --run-diffusion \
  --num_inference_steps 20 \
  --lowfreq_alpha 0.8 \
  --lowfreq_cutoff 0.25 \
  --lowfreq_order 4 \
  --lowfreq_protect_fg \
  --lowfreq_mask_feather 4.0
```
object 폭을 배경 폭의 4.5%~8.5%로 샘플링

생성 결과:

```text
diffusion_result/result/
  ch002_object_1_0000001.jpg
  ch002_object_2_0000002.jpg
```

## 8. 이미 생성된 Input으로 Diffusion만 실행

```bash
cd /path/to/Tale
CUDA_VISIBLE_DEVICES=1 python posco_main.py \
  --model_path stable-diffusion-2-1-base \
  --data_dir diffusion_result/input \
  --output_dir diffusion_result/result \
  --flat_output \
  --output_ext .jpg \
  --default_prompt "a cardboard box on a railway track at a steel mill" \
  --num_inference_steps 20 \
  --crop_padding 0.5 \
  --lowfreq_alpha 0.8 \
  --lowfreq_cutoff 0.25 \
  --lowfreq_order 4 \
  --lowfreq_protect_fg \
  --lowfreq_mask_feather 4.0 \
  --skip-existing
```


