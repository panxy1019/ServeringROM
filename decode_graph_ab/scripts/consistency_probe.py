#!/usr/bin/env python3
import argparse, asyncio, hashlib, json, random, time
import httpx


async def main():
    p = argparse.ArgumentParser(); p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", default="qwen36-27b-w8a8"); p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--count", type=int, default=8); args = p.parse_args()
    rng = random.Random(args.seed)
    prompts = [" ".join([f"事实{rng.randrange(10000)}" for _ in range(500)]) + "\n请只续写一段。" for _ in range(args.count)]
    semaphore = asyncio.Semaphore(2)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=600) as client:
        async def one(i, prompt):
            t = time.perf_counter()
            async with semaphore:
                response = await client.post("/v1/completions", json={
                    "model": args.model, "prompt": prompt, "max_tokens": 64,
                    "temperature": 0, "seed": args.seed + i, "ignore_eos": True})
            body = response.json(); text = body.get("choices", [{}])[0].get("text", "")
            return {"index": i, "status": response.status_code, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text": text, "usage": body.get("usage", {}), "elapsed_seconds": round(time.perf_counter()-t, 3)}
        rows = await asyncio.gather(*(one(i, x) for i, x in enumerate(prompts)))
    print(json.dumps({"seed": args.seed, "count": len(rows), "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__": asyncio.run(main())
