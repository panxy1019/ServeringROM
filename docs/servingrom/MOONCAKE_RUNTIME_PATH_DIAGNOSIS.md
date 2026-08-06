# Ascend Mooncake 运行时路径诊断

## 结论

冻结 D2 基线实际执行的 connector 不是 vLLM 通用实现，而是 vLLM-Ascend 插件注册的：

```text
connector: vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.MooncakeConnector
worker:    vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.MooncakeConnectorWorker
thread:    vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector.KVCacheRecvingThread
engine:    mooncake.engine.TransferEngine
method:    batch_transfer_sync_read
owner:     Decode TP worker
```

源码文件为：

```text
/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
```

Decode 采用 pull 语义：每个 Decode TP rank 根据真实 `src_list/dst_list/length_list`
调用 `batch_transfer_sync_read()`，从 Prefill 暴露的 KV 地址读取到 Decode KV cache。
Prefill 侧负责注册内存和提供 metadata，不在请求主路径执行对称的 Python send 调用。

## 运行时证据

Ascend 插件注册表明确映射：

```text
MooncakeConnectorV1
  -> vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector
  -> MooncakeConnector
```

原 D2 Decode 日志中的消息：

```text
KV cache transfer for request ... took ... local_device_id ... remote_session_id ...
```

位于 vLLM-Ascend connector 第 826 行附近；该字符串不存在于运行镜像的通用 vLLM
connector。Python 运行时检查也确认通用类的代码对象仍指向 `/vllm-workspace/vllm/...`，
而日志与调用来自 `/vllm-workspace/vllm-ascend/...`。

新镜像启动后，每个实际 Decode TP worker 还会生成一次 capability marker，保存：

```text
connector / worker / TransferEngine / method / source_file
process_id / engine_id / tp_rank / tp_size / role
```

该 marker 是最终运行时证明，避免只凭静态注册表推断。

## 旧 Hook 未命中的原因

旧补丁修改的是：

```text
vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector
```

在 Ascend 插件激活后，`MooncakeConnectorV1` 由 vLLM-Ascend 自己注册并实例化。两者类名
相似，但模块、线程模型和传输语义不同：通用实现包含 sender/pull 协程，而当前 Ascend
实现由 Decode `KVCacheRecvingThread` 调用 CANN 环境中的 Mooncake TransferEngine。
因此旧 Python hook 可成功构建，却永远不在实际请求路径执行。

## 最小 Hook 边界

新 hook 只位于三个边界：

1. `add_request()` 入队后记录 enqueued；
2. `length_list` 完成后、`batch_transfer_sync_read()` 之前记录 started；
3. TransferEngine 返回或抛错后记录 completed/failed。

没有增加 `torch_npu.synchronize()`、NPU synchronize、KV buffer 复制、额外 transfer、
请求重排或队列等待。`emit()` 仍只执行轻量封装和非阻塞 `put_nowait()`。

## 字节语义

`actual_bytes = sum(length_list)` 是 TransferEngine 本次实际收到的传输描述符长度总和。
该值在 Decode 进程观察，但数据源属于 `source_engine` 指向的 Prefill KV 注册区；因此报告
同时保留 `transfer_role=receive` 与 source/target engine，避免把 pull 执行者误写成发送者。
