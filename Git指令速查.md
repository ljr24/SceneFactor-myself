# Git 指令速查（李嘉睿版）

按使用频率分类，总结自项目协作实际场景。

---

## 一、配置

### 1.1 首次安装配置

```bash
git config --global user.name "你的名字"      # 设置用户名
git config --global user.email "你的邮箱"     # 设置邮箱
```

- `--global` 表示全局生效（这台机器所有仓库都用这个配置）
- 不加 `--global` 则只对当前仓库生效
- 查看已有配置：`git config --global user.name`

### 1.2 SSH Key（免密连接 GitHub）

```bash
ssh-keygen -t ed25519 -C "你的邮箱"          # 生成密钥（一路回车）
cat ~/.ssh/id_ed25519.pub                   # 复制公钥 → 贴到 GitHub Settings
ssh -T git@github.com                       # 验证是否配置成功
```

- **私钥** `~/.ssh/id_ed25519` — 自己保管，不要泄露
- **公钥** `~/.ssh/id_ed25519.pub` — 给 GitHub，可以公开
- 每次配新电脑都要做一次这个

---

## 二、本地仓库操作

### 2.1 初始化

```bash
git init            # 把当前目录变成 Git 仓库
```

- 执行后会在当前目录生成一个 `.git/` 文件夹（里面存了所有版本历史）
- 一个项目只初始化一次

### 2.2 查看状态

```bash
git status          # 查看当前文件状态（最常用的命令！）
```

- 红字 = 未被跟踪（untracked）/ 被修改但还没暂存
- 绿字 = 已暂存（staged），准备提交

### 2.3 暂存与提交

```bash
git add <文件名>                  # 暂存某个文件
git add .                        # 暂存所有变更
git rm --cached <文件名>          # 取消暂存（从暂存区移除）

git commit -m "提交说明"          # 提交已暂存的文件
git commit -am "说明"             # 跳过 add，直接暂存+提交所有已跟踪的文件
```

**三区概念**：
```
工作区（你看到的文件） → git add → 暂存区（Staging） → git commit → 本地仓库
```

- `commit` 是本地操作，不涉及网络
- 提交说明要写清楚这次做了什么

### 2.4 查看历史

```bash
git log                          # 查看提交历史
git log --oneline                # 一行一个提交，更简洁
```

---

## 三、远程仓库操作

### 3.1 关联远程仓库

```bash
git remote add origin <地址>      # 添加远程仓库，别名 origin
git remote -v                    # 查看已关联的远程地址
git remote set-url origin <地址>  # 修改远程地址
git remote add team <地址>        # 添加第二个远程地址，别名 team
```

- `origin` 是默认别名，可以改成任何名字
- 常用 SSH 地址格式：`git@github.com:用户名/仓库名.git`
- 同一份代码可以推送到多个远程仓库

### 3.2 推送（Push）

```bash
# 首次推送（需要关联上游分支）
git push -u origin main

# 以后推送（只要 1 行）
git push

# 推送到不同远程
git push origin main              # 推到自己仓库
git push team main               # 推到导师项目
```

- `-u` = `--set-upstream`，建立本地分支和远程分支的关联，第一次推时用
- `main` 是当前分支名

### 3.3 拉取（Pull）

```bash
git pull                         # 拉取远程最新代码并合并到本地
git pull origin main             # 从指定远程拉取

# 如果远程有文件而本地也有修改：
git pull --rebase                # 用 rebase 方式拉取，避免产生合并节点
```

**工作流程**：改代码前先 `git pull`，避免冲突。

### 3.4 克隆（Clone）

```bash
git clone <地址>                  # 下载一个完整的仓库到本地
git clone git@github.com:AkiraJM/SceneFactor.git
```

- 会自动把远程仓库设成 `origin`
- 不需要再 `git init`

---

## 四、分支操作

### 4.1 基础分支操作

```bash
git branch                       # 查看本地所有分支（当前分支前有 *）
git branch -a                    # 查看所有分支（含远程）
git branch <分支名>               # 创建新分支
git checkout <分支名>             # 切换到另一个分支
git checkout -b <分支名>          # 创建并切换到新分支（最常用）
git branch -m <新名字>            # 重命名当前分支
```

### 4.2 合并与删除

```bash
git merge <分支名>                # 把指定分支合并到当前分支

# 先回到 main，再把 feature 合并进来
git checkout main
git merge feature

git branch -d <分支名>            # 删除本地分支（已合并的）
git branch -D <分支名>            # 强制删除本地分支（没合并也删）
```

---

## 五、撤销与回退

```bash
git restore <文件名>              # 撤销工作区的修改
git restore --staged <文件名>     # 把文件从暂存区移回工作区（等价于 git rm --cached）
git reset --soft HEAD~1          # 撤销最近一次 commit，但保留修改（重新提交用）
git reset --hard HEAD~1          # 撤销最近一次 commit，不保留修改（慎用！）
```

- `HEAD` = 当前所在的提交
- `HEAD~1` = 上一个提交，`HEAD~2` = 上两个，以此类推
- `--hard` 会丢失修改，确认清楚再用

---

## 六、协作流程（以本项目为例）

### 推到自己仓库练手

```bash
# 关联远程仓库
git remote add origin git@github.com:ljr24/SceneFactor-myself.git
git branch -m main
git add .
git commit -m "first commit"
git push -u origin main
```

### 后续日常更新

```bash
git add <你要改的文件>
git commit -m "fix: 修复 Step 4 内存溢出问题"
git push
```

### 推到导师项目（拿到权限后）

```bash
# 方式一：直接换 origin 地址
git remote set-url origin git@github.com:AkiraJM/SceneFactor.git
git push

# 方式二：保留自己仓库，新增 team 远程
git remote add team git@github.com:AkiraJM/SceneFactor.git
git push team main
```

---

## 七、完整工作流示意图

### 单人开发

```
git pull                    ← 先拉取最新
  ↓
改代码
  ↓
git add .                   ← 暂存
git commit -m "说明"         ← 提交到本地
git push                    ← 推送到远程
```

### 多项目切换

```
git remote add team 导师项目    ← 加导师仓库
git remote add hf  huggingface  ← 加HF仓库

git push team main            ← 推到导师项目
git push hf main              ← 推到HuggingFace
git push origin main          ← 推到自己仓库
```

---

## 八、常见报错与解决

| 报错信息 | 原因 | 解决 |
|---------|------|------|
| `Please tell me who you are` | 没配用户名邮箱 | `git config --global user.name/email` |
| `Permission denied (publickey)` | SSH Key 没配好 | 重新 `ssh-keygen` + 加到 GitHub |
| `Invalid username or token` | HTTPS 方式但用了密码 | 改用 SSH 地址 |
| `Repository not found` | 仓库不存在/没权限 | 检查地址 + 问导师加权限 |
| `failed to push` | 远程有本地没有的文件 | `git pull --rebase` 再重试 |
| `source ref spec main has no match` | 还没 commit 就 push | 先 `git commit` |
| `Authentication failed` | 密码/token 不对 | 用 SSH 方式替代 HTTPS |

---

*整理于 2026-05-27，按实际使用场景分类*
