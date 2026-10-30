# 项目协作与上游追踪

## 权威来源

| 组件 | 追踪仓库 | 职责与同步原则 |
| --- | --- | --- |
| 开源 Tributo Core | https://github.com/jiangxt2/Tributo | 开源版的权威上游。同步其最新默认分支，保留尚未被上游吸收的本地功能；已被吸收的实现采用上游最新版 |
| ClickHouse 读取连接器 ray-clickhouse | https://github.com/jiangxt2/ray-clickhouse | ClickHouse 读取、过滤下推和连接器原生接口的权威来源。以该仓库的公开 API 为准，不把内部旧适配或本地补丁当成最新接口 |
| Redis Broker | https://github.com/Chang-Tong/tributo-broker-redis | Broker 实现及 external 协议落地、契约测试和 Redis/Tributo 反馈校验的主仓库 |

“追踪”表示更新与兼容性判断以以上仓库为准，不表示构建时自动使用不受控的 HEAD。代码、依赖、镜像和 CI 应锁定已验证的版本或完整提交 SHA。

## 同步与兼容

- 先检查本地分支、未提交改动和远端地址，再 fetch 上游；不要通过 reset、强推或覆盖未提交文件来同步。
- 比较上游变化后合并，保留用户的计划、文档和尚未被上游覆盖的工作。冲突按最新公开接口处理；失效接口直接淘汰，不新增旧 API 兼容壳。
- 连接器升级时同时检查 Core 的读取绑定、过滤下推及相关写入绑定调用；以公开连接器接口为边界，不在 Core 重复实现数据面能力。
- Core 升级后验证 Broker 的请求映射、当前 Core 结果对象、序列化反馈及真实 Redis 契约测试。不能因旧环境中的测试通过而宣称新 Ray/连接器运行环境兼容。
- 兼容性通过后才更新下游的 Core SHA、连接器版本、锁文件和部署依赖；同步维护来源、补丁和 Wheel 哈希。明确区分本地更新、已推送代码、远端 CI 和真实集群验收。
- external 协议实现与共用契约测试由 Broker 承接，KnoVa 保持集成薄层，不再向 Core 或 KnoVa 复制同一套协议实现。

## 开发与验收

遵循 `CONTRIBUTING.md`、`docs/developer/testing.md`，测试策略以 `ci/test-suites.json` 和 `scripts/ci_test_plan.py` 为准。使用 uv.lock 与声明的 extras 准备环境，不隐式替换已验证的依赖组合。

交付时必须核对：格式、lint、类型检查、禁止新增忽略注释、相关单元测试、包入口发现和 Wheel 构建。具体命令遵守本仓库 CI 与 precheck；不得通过关闭检查、删除断言或新增忽略注释绕过失败。

- 格式：`uv run --locked --no-sync ruff format --check .`。
- Lint：`uv run --locked --no-sync ruff check .`。
- 测试清单审计：`python scripts/ci_test_plan.py audit`；通过 planner 选择并通过受控 runner 执行相关 suite。
- Review 或推送前完整预检：`uv run --locked --no-sync python scripts/pr-precheck.py`；先准备需要的解释器和依赖，`--skip-tests` 不算完整通过。
- Wheel 构建：`uv build --wheel`；检查构建产物的版本、公开导入和入口发现。
- 跨仓 Broker 检查还应遵循 Broker 自己的 `AGENTS.md`，包括独立协议门禁及真实 Redis 证据要求。

检查被跳过、进程非零退出、缺少真实依赖或证据，都不能报告为通过。模拟执行、Schema 检查与历史回放不能替代真实 Ray/数据库端到端验收。
