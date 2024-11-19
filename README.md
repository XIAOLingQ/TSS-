```
python -m llava.serve.controller --host 0.0.0.0 --port 20000
python -m llava.serve.gradio\_web\_server --controller http://localhost:20000 --model-list-mode reload
python -m llava.serve.model\_worker --host 0.0.0.0 --controller http://localhost:20000 --port 40000 --worker http://localhost:40000 --model-path "/root/autodl-tmp/llava/"
```
