docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
  -v C:\Users\bertr\PycharmProjects\ASR_Agent:/workspace `
  -w /workspace `
  xavier/asr-agent:dev-pc python scripts/run_asr_pipeline.py tests/data/test.mp3 --out outputs --lang fr --ns 1 --denoise