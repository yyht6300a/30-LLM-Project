# model_runtime：vLLM 与 logprobs 通道

所有者：P3。

`vllm/` 下提供本地 Qwen2.5-7B-Instruct 启动脚本与联通性测试，确认
choices[0].logprobs.content 可取、思考已关、解码与训练一致。
