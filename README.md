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
