# mcp_server：校准器 MCP 服务

所有者：P3。

暴露 calibrate(question, answer, c) 工具，内部完成实体链接、查流行度表、组装特征、加载
权重、MLP 推理，返回校准概率与 decision。缺权重、缺特征或模型不匹配时降级返回 unknown 或
原始 c，不阻塞 agent。
