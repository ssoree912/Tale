  CUDA_VISIBLE_DEVICES=1 python posco_rail_diffusion.py \
    --train-dir /정상데이터/폴더/경로 \
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
    --num_inference_steps 20

# 이미 생성된 input에서 전체 약 1000개 균등 diffusion

```bash
CUDA_VISIBLE_DEVICES=1 python posco_main.py \
  --model_path stable-diffusion-2-1-base \
  --data_dir diffusion_result/input \
  --output_dir diffusion_result/result \
  --flat_output \
  --output_ext .jpg \
  --default_prompt "a cardboard box on a railway track at a steel mill" \
  --num_inference_steps 20 \
  --crop_padding 0.5 \
  --target-sample-count 1000 \
  --skip-existing
```

# 배경 있는 실제 물체 사진 + segmentation mask 사용

`objects` 폴더에 실제 물체 사진과 같은 stem의 mask를 같이 두면 자동 배경 제거 대신 mask를 사용합니다.

```text
objects/object_box.jpg
objects/object_box_mask.png
```

지원 mask 이름 예시:

```text
object_box_mask.png
object_box_seg.png
object_box_segmentation.png
```

같은 물체를 여러 위치/크기로 보여주려면 input 생성 때 placement 옵션을 조절합니다.

```bash
CUDA_VISIBLE_DEVICES=1 python posco_rail_diffusion.py \
  --train-dir /정상데이터/폴더/경로 \
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
  --placement-size-frac 0.035 0.070 \
  --large-placement-size-frac 0.080 0.130 \
  --large-placement-every 2 \
  --placements-per-pair 2 \
  --overwrite
```
