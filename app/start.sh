#!/bin/bash
# 启动 pair_construction Streamlit 看板（inspire 端）
#
# 用法（inspire 上）：
#   bash app/start.sh                # 默认端口 8501
#   PORT=18501 bash app/start.sh     # 改端口
#
# 本机访问（SSH 转发）：
#   ssh -L 8501:localhost:8501 qz_zxy_gpu_4090
#   然后浏览器打开 http://localhost:8501
set -u

# 复用 kxhuang 的 tts env（已装好 streamlit + plotly + pyarrow）
PYBIN=${PYBIN:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/miniconda/envs/tts/bin/python}
STREAMLIT=${STREAMLIT:-/inspire/hdd/project/embodied-multimodality/public/kxhuang/miniconda/envs/tts/bin/streamlit}
PORT=${PORT:-8501}

PROJ_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJ_ROOT"

# 索引存在性提示（不强制阻断）
if [ ! -f "$PROJ_ROOT/app/data/index.parquet" ]; then
    echo "[start] 警告：app/data/index.parquet 不存在，Dashboard 会空白。"
    echo "[start] 建议先跑： $PYBIN app/index_builder.py"
fi
if [ ! -f "$PROJ_ROOT/app/data/raw_source.parquet" ]; then
    echo "[start] 警告：app/data/raw_source.parquet 不存在。"
    echo "[start] 建议先跑： $PYBIN app/raw_scanner.py --add instruction_0.1_enzh:zh:<dir> --add instruction_0.1_enzh:en:<dir>"
fi

echo "[start] streamlit on port $PORT"
echo "[start] SSH tunnel from 本机: ssh -L $PORT:localhost:$PORT qz_zxy_gpu_4090"
echo "[start] 浏览器: http://localhost:$PORT"
# 用 python -m streamlit 而不是 streamlit shebang，避免 PYTHONPATH/PYTHONHOME 污染导致 plotly 等找不到
exec "$PYBIN" -m streamlit run "$PROJ_ROOT/app/app.py" \
    --server.port "$PORT" \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.enableWebsocketCompression false \
    --browser.gatherUsageStats false
