FROM quay.io/ascend/vllm-ascend:v0.22.1rc1-a3

ARG PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple

RUN python3 -m pip install --no-cache-dir \
      --index-url "${PIP_INDEX_URL}" \
      "ray[default]==2.48.0" \
      "httpx>=0.27,<1" \
      "psutil>=5.9,<8" \
    && python3 -c "import mooncake, ray; print('ray', ray.__version__, 'mooncake', mooncake.__file__)" \
    && test -x /usr/local/bin/mooncake_master \
    && test -f /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py

LABEL org.opencontainers.image.title="Qwen3.6-27B W8A8 PD learning worker"
LABEL org.opencontainers.image.description="vLLM-Ascend 0.22.1rc1 A3, Mooncake, Ray 2.48.0"

