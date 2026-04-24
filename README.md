# AIVC - Monorepo大单仓项目

使用 Python 构建的全栈 AI 对话系统，采用 Monorepo 架构管理。

## 📁 项目结构

```
AIVC/
├── apps/
│   └── frontend/        # Gradio 前端界面
│       ├── app.py       # Gradio 主应用
│       └── requirements.txt
├── package.json         # pnpm 工作区配置
├── pnpm-workspace.yaml
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 使用 uv 创建虚拟环境（推荐）
uv venv .venv
source .venv/bin/activate
# 安装 Python 依赖
uv pip install -r apps/frontend/requirements.txt
```

### 2. 启动服务
终端- 启动前端:
```bash
PYTHONUNBUFFERED=1 .venv/bin/python apps/frontend/app.py
# 前端运行在 http://0.0.0.0:7857
```

### 3. 调试卡顿（查看步骤日志）

前端默认会打印阶段日志（上传、DCT网格、块重建进度、总耗时）。  
如需关闭前端调试日志：

```bash
AIVC_DEBUG=0 .venv/bin/python apps/frontend/app.py
```

为避免日志缓冲，建议加 `PYTHONUNBUFFERED=1`：

```bash
PYTHONUNBUFFERED=1 pnpm dev
```

## 🔧 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | Gradio 4+（已验证 6.x） | AI 友好的 Web 界面框架 |
| 架构 | Monorepo | 使用 pnpm workspaces 管理 |


## 🎨 功能特性

- **智能对话**: 与后端 AI 服务实时交互
- **状态监控**: 实时显示后端连接状态
- **API 测试**: 内置 API 调试工具
- **响应式界面**: 适配不同屏幕尺寸

## 📝 开发说明

### 添加新的子项目

1. 在 `apps/` 目录下创建新项目
2. 添加 `package.json` 文件
3. 项目会自动被 pnpm workspace 识别



## 📄 License

MIT
