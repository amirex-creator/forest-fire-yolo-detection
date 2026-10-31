# GitHub + Cloudflare 部署说明

本项目的网页入口是 `app.py`。GitHub 负责保存代码和模型，Cloudflare Tunnel 负责把正在运行的 Streamlit 服务暴露到公网；YOLO 推理仍由运行 Python、PyTorch 和 Ultralytics 的服务器完成。

## 1. 发布内容

仓库只需要包含源代码、部署配置、报告图片和以下两个演示模型：

```text
models/hardneg_finetune_cont30_full_20261029_034435_best.pt
models/batch1_e10_full_20261028_130129_best.pt
```

完整数据集、虚拟环境、训练运行目录和 `便携版/` 不应进入公开仓库。项目中的 `.gitignore` 和 `.dockerignore` 已按此规则配置。数据集来源和许可边界见 `README.md` 与 `data/数据准备说明.md`。

## 2. 本机临时公网演示

### 启动 Streamlit

在项目根目录运行：

```powershell
$env:FOREST_SMOKE_PUBLIC_MODE = "1"
$env:FOREST_SMOKE_DEVICE = "cpu"
$env:FOREST_SMOKE_MODEL = "models/hardneg_finetune_cont30_full_20261029_034435_best.pt"
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

也可以双击项目根目录的 `run_streamlit_public.bat`。脚本会优先寻找项目内或上级目录的 Python 虚拟环境，找不到时使用系统 `python`。如果 8501 已被旧服务占用，它会自动改用 8502，并把端口写入 `.forest_smoke_public_port`，供 Tunnel 脚本读取。

### 启动 Quick Tunnel

另开终端运行：

```powershell
cloudflared tunnel --url http://127.0.0.1:8501
```

终端会输出一个 `https://*.trycloudflare.com` 地址。把该地址发给老师即可访问。

Quick Tunnel 的限制：本机必须开机、Streamlit 和 Tunnel 进程都必须保持运行，电脑休眠或断网后地址不可用，地址可能随下次启动变化。`start_cloudflare_tunnel.bat` 会执行同一命令，但当前电脑需要先安装 `cloudflared` 并加入 PATH。

## 3. 固定域名 Tunnel

固定网址需要 Cloudflare 账号和已经接入 Cloudflare 的域名。在登录 Cloudflare 后执行：

```powershell
cloudflared tunnel login
cloudflared tunnel create forest-smoke
cloudflared tunnel route dns forest-smoke forest-smoke.example.com
cloudflared tunnel run forest-smoke
```

把 `deploy/cloudflared/config.example.yml` 复制为本机 Cloudflare 配置文件，并替换 Tunnel UUID、凭证路径和域名。凭证文件只能保存在本机，不得提交到 GitHub。

固定 Tunnel 仍然要求运行 Streamlit 的机器持续在线。如果希望电脑关机后仍可访问，应把 Docker 服务迁移到云服务器或其他长期运行的平台，再在该平台运行 `cloudflared`。

## 4. Docker / 云服务器

`Dockerfile` 使用 CPU 推理，默认模型为最终 40 轮模型：

```powershell
docker build -t forest-smoke-web .
docker run --rm -p 8501:8501 forest-smoke-web
```

服务器上可以让 Cloudflare Tunnel 指向 `http://127.0.0.1:8501`。CPU 推理速度取决于服务器配置；视频检测会占用更多 CPU、内存和磁盘，应限制演示视频长度。

Docker 当前未安装。官方 `cloudflared` 已下载到工作区外的 `E:\forest_smoke\tools\cloudflared.exe` 并验证版本；但本机网络策略禁止它直接连接 Cloudflare 边缘，因此本次未取得真实 `trycloudflare.com` 地址。安装工具并允许 Cloudflare 出站连接后，再执行实际 Tunnel 部署。

## 5. GitHub 仓库

GitHub 公共仓库已经创建：`https://github.com/amirex-creator/forest-fire-yolo-detection`。本地提交已完成，当前提交为 `133d719`；如果命令行凭证可用，在项目根目录运行：

```powershell
git init -b main
git add .
git status --short
git commit -m "prepare forest fire detection web deployment"
git remote add origin https://github.com/<用户名>/<仓库名>.git
git push -u origin main
```

本次机器环境的 Git 凭证存储不可用，且自动化浏览器授权被网络审核拦截，因此仓库创建后仍需在有 GitHub 凭证的终端执行一次上面的推送命令。推送成功后，仓库页面会显示完整代码和模型。

也可以双击项目根目录的 `push_to_github.bat`。它会在可见窗口中运行推送，避免命令静默等待认证；认证完成后重新运行即可。

推送前确认 `git status --short` 中没有数据集、虚拟环境、便携版、Cloudflare 凭证或个人文件。两个演示模型允许直接进入仓库；更多模型建议放到 GitHub Release 或模型仓库。

## 6. Cloudflare Pages 的边界

Cloudflare Pages 可以托管项目介绍页和静态文件，但不能直接运行当前 Python、PyTorch、Ultralytics 或 Streamlit 应用。不要把 Pages 部署成功误认为 YOLO 推理服务已经上线。当前项目应使用 Cloudflare Tunnel，或者使用能运行 Docker/Python 的云服务器，再通过 Cloudflare 提供域名和访问入口。

## 7. 公网运行限制

- 公网模式默认使用 CPU，并通过 `FOREST_SMOKE_DEVICE=cpu` 固定设备。
- 上传图片默认不超过 20 MB，视频默认不超过 120 MB。
- 公网上传和结果写入 `outputs/predictions/public/`，默认保留 24 小时后清理。
- 不要上传个人隐私、敏感位置或需要保密的图片。
- 结果仅供课程演示和辅助分析，不能作为真实火灾报警或安全决策依据。
