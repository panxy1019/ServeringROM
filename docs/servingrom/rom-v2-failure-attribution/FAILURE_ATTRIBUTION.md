# ServingROM v2 失败归因

- 输入：Dataset v1.1 的 `201600` 个已封存窗口；没有重新采集或运行模型。
- v1.1 manifest：`e5bc4802c232b53b549c4a1eb003837af378d726af05d98e3da65674d668cf1e`；源 Dataset v1 manifest：`2b83e758aaec7b1ff814f7afb8585d052921e5dc0fff294f83f408532feb2518`。
- 根因一：200 ms 的 Y 是完成事件流，在低请求率下高度零膨胀；状态库存与同窗完成流之间还存在服务时间延迟。
- 根因二：v1 POD 只优化状态重构能量，未保证 goodput、TTFT、TPOT 所需的低能量方向进入 reduced state。
- 根因三：v1 的一阶状态没有显式速度和上一窗口到达记忆，难以区分相同库存下的积压与恢复方向。
- 采样尺度带来的 validation 关键输出 NRMSE 改善：`0.319775`。
- 仅增加 200 ms 短期记忆带来的改善：`0.008280`。
- 状态动力学候选：period=200 ms, rank=16, memory=True, validation state NRMSE=0.677922。
- 输出观测候选：period=5000 ms, rank=16, memory=True, validation key-output NRMSE=0.466693。
- 单一时钟同时通过状态与关键输出门：`False`；因此采用多速率设计。
- 慢速输出头关键输出 held-out NRMSE：`{"validation": {"completed_output_tokens": 0.5815433175457487, "goodput_output_tokens": 0.5967627571030731, "ttft_sum_ms": 0.6263985312334097, "tpot_sum_ms": 0.5697187601922329, "kv_transfer_completed_bytes": 0.32410809719422845, "prefill_scheduled_tokens": 0.14367309388701663, "decode_scheduled_tokens": 0.4246469301595327}, "test": {"completed_output_tokens": 0.5826903899308488, "goodput_output_tokens": 0.6120158286498418, "ttft_sum_ms": 0.6121380301495712, "tpot_sum_ms": 0.5837580283378835, "kv_transfer_completed_bytes": 0.34990177792705796, "prefill_scheduled_tokens": 0.15067665630569077, "decode_scheduled_tokens": 0.42336265291595554}, "test/transient": {"completed_output_tokens": 0.6010176532847199, "goodput_output_tokens": 0.6276839315605021, "ttft_sum_ms": 0.654454931988107, "tpot_sum_ms": 0.6114883729747954, "kv_transfer_completed_bytes": 0.39677950172969073, "prefill_scheduled_tokens": 0.18486015204129488, "decode_scheduled_tokens": 0.44479878216401014}}`。
- workload、load fraction、arrival process 与 transient pattern 的分解位于 `audit/error_attribution_by_run.json`。
