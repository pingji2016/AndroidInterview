### 6月1日（周一）Day 8
| 时段 | 任务 | 产出 |
|------|------|------|
| 30min | Activity 启动模式：standard、singleTop、singleTask、singleInstance | 画出每种模式的任务栈变化图 |
| 30min | flags 和 taskAffinity 的作用 | 写一个场景：如何让 Activity 在新的任务栈中启动 |
| 60min | 手写：从 A 启动 B（singleTask），画出栈的变化 | 一张图 |


## Activity 启动模式笔记

### standard（标准模式）
- 每次启动都创建新实例，压入当前任务栈
- 谁启动它就进入谁的栈（即发起者的任务栈）
- 典型的 **覆盖压栈** 行为

### singleTop（栈顶复用）
- 栈顶已存在该 Activity 实例 → 复用，调用 `onNewIntent()`
- 栈顶不存在 → 创建新实例（同 standard）
- 适用于：推送通知详情页、搜索页面（避免栈顶重复）

### singleTask（栈内复用）
- 栈内已存在该 Activity 实例 → 清除其之上的所有 Activity，复用，调用 `onNewIntent()`
- 栈内不存在 → 创建新实例
- 会检查 `taskAffinity` 决定是否创建新栈
- 适用于：主页面（Home）、应用入口页

### singleInstance（单栈单实例）
- 独占一个任务栈，栈内只有它自己
- 启动其他 Activity 时会在其他栈中打开
- 适用于：来电界面、系统级弹窗（需要与 App 完全隔离的场景）

### 关键对比

| 模式 | 创建新实例 | 复用时机 | 任务栈行为 |
|:---|:---|:---|:---|
| standard | 总是 | 不复用 | 进入调用者栈 |
| singleTop | 栈顶不存在时 | 栈顶已存在 | 进入调用者栈 |
| singleTask | 栈内不存在时 | 栈内已存在（清空其上） | 根据 taskAffinity 决定 |
| singleInstance | 首次启动 | 已存在实例 | 独占独立栈 |

### flags 相关
- `FLAG_ACTIVITY_NEW_TASK`：在新的任务栈中启动（配合 taskAffinity）
- `FLAG_ACTIVITY_CLEAR_TOP`：清除目标 Activity 之上的页面
- `FLAG_ACTIVITY_SINGLE_TOP`：同 singleTop 效果
- `FLAG_ACTIVITY_CLEAR_TASK`：启动前清空整个栈（配合 NEW_TASK 使用）

### taskAffinity 的作用
- 默认值与包名相同
- 与 `FLAG_ACTIVITY_NEW_TASK` 或 `singleTask`/`singleInstance` 配合生效
- 不同 Affinity 的 Activity 会进入不同的任务栈
- 应用场景：让某个页面在最近任务列表中独立显示