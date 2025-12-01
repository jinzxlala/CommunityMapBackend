# CommunityMapBackend

桐梓林国际社区电子地图项目的后端服务，基于 **Django + Django REST framework** 实现，主要负责：

- **管理社区地标数据（Location）**：名称、经纬度、描述、图片、分类、创建者、收藏关系等；
- **对外提供 REST API**：为前端 `index.html` 的地图页面提供地标列表、详情、增删改等接口；
- **对接图片识别模块**：接收上传的门头照片，调用 `image_detection_package`，尝试识别对应地标；
- **基础认证与权限**：区分普通用户和管理员，控制谁能创建、编辑、删除地标。

详细的汇报背景和整体方案可参考仓库根目录下的 `开源年会汇报准备方案.md`。

---

## 核心功能概览

- **地标数据管理（Location）**
  - 存储地标的经纬度、名称、描述、图片、分类等信息；
  - 通过 `owner` 记录创建者，通过 `favorites` 记录收藏该地标的用户；
  - 所有地标会以 GeoJSON 风格返回给前端，方便直接绘制在 Mapbox 地图上。

- **RESTful API 接口**
  - 地标接口：
    - `GET /api/locations`：获取所有地标列表；
    - `GET /api/locations/<id>`：获取单个地标详情；
    - `POST /landmarks`（兼容前端）：创建新地标（管理员权限）；
    - `PUT /landmarks/<id>` / `DELETE /landmarks/<id>`：更新、删除地标（管理员权限）。
  - 图片上传与识别：
    - `POST /api/upload-image/`：上传图片，调用 `image_detection_package` 进行识别，并尝试匹配数据库中的 `Location`。
  - 用户与收藏：
    - 通过 `favorites` 多对多关系，支持用户收藏地标，并在前端显示“我的收藏”。
  - 认证相关：
    - `auth/login`、`auth/logout`、`auth/check` 等接口，用于登录、退出和检查登录状态。

> 更详细的接口说明，请查看：`api-document.md`。

---

## 技术栈

- **语言与框架**
  - Python 3.x
  - Django
  - Django REST framework

- **依赖（见 `requirements.txt`）**
  - `django`, `djangorestframework`, `django-filter`, `django-cors-headers`
  - `Pillow`（图片处理）
  - `requests`, `dashscope`（调用通义千问多模态等云端识别服务）

- **数据库**
  - 默认使用 SQLite（开发环境：`db.sqlite3`），可根据需要切换到 MySQL/PostgreSQL。

---

## 本地开发环境搭建

在项目根目录 `CommunityMapBackend/` 下进行以下操作。

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell 可使用: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2. 数据库迁移

```bash
python manage.py migrate
```

如需创建管理员账号（用于登录后台或管理员接口）：

```bash
python manage.py createsuperuser
```

### 3. 启动开发服务器

```bash
python manage.py runserver 0.0.0.0:8000
```

启动后，可以访问：

- `http://localhost:8000/`：前端可接入的根路由（当前主要由外部 `index.html` 使用 API）；
- `http://localhost:8000/admin/`：Django 管理后台；
- `http://localhost:8000/api/locations`：地标列表 API 示例。

> 注意：前端地图页面 `index.html` 当前位于仓库根目录，通常通过静态服务器或本地打开，并将 `API_BASE_URL` 指向该后端服务地址。

---

## 图片识别模块集成说明

图片识别相关逻辑位于 `image_detection_package/`，通过接口与后端视图对接：

- 主要入口函数在 `image_detection_package/recognition_dispatcher.py` 中封装，并在 `__init__.py` 中导出：
  - `get_landmark(image_path)`：综合通义千问多模态和本地模型的结果，返回识别出的地标名称；
  - `get_landmark_qwen(...)`：仅使用云端 Qwen；
  - `get_landmarks_qwen_in_dir(...)`：对目录中的图片批量识别；
  - `get_landmark_minimal(...)`：更简化的调用方式。
- 后端视图 `ImageUploadView` 会：
  1. 接收前端上传的图片并保存到 `media/` 目录；
  2. 调用上述识别函数获取候选地标名称；
  3. 在数据库中尝试匹配对应的 `Location`；
  4. 将图片 URL + 识别结果一起返回给前端。

---

## 数据导入 / 导出与运维

为方便维护社区地标数据，项目提供了管理命令（位于 `myapp/management/commands/`）：

- 批量导入：`import_locations.py`  
  - 支持从 CSV 批量导入地标（如 `地标详情表(校正).csv`），快速初始化或更新数据。
- 批量导出：`export_locations.py`  
  - 将当前数据库中的所有 `Location` 导出为 CSV，用于备份或与外部工具对比。

配合 `coordinate_analysis_report.md` 中的实验，可对比 AI 生成坐标与人工/地图服务坐标的偏差，从而制定数据质量策略：

- **结论**：AI 可用于生成文字描述，但地理坐标应以高德地图等专业服务和人工校正为准。

---

## 目录结构（简要）

```text
CommunityMapBackend/
├── manage.py
├── requirements.txt
├── db.sqlite3
├── index.html                 # 前端地图单页（示例/联调用）
├── api-document.md            # 后端接口文档
├── coordinate_analysis_report.md
├── coordinate_comparison_report.txt
├── 地标详情表(校正).csv
├── myapp/
│   ├── models.py              # Location 等数据模型
│   ├── views.py               # 核心视图与 API
│   ├── urls.py                # 应用级路由
│   ├── serializers.py         # GeoJSON 序列化等
│   └── management/
│       └── commands/          # import/export 脚本
├── map/
│   └── urls.py                # 项目级路由
└── image_detection_package/   # 图片识别集成（Qwen + 本地模型）
```

---

## 参考与致谢

本项目在实现过程中依赖并感谢以下开源项目和服务：

- Django, Django REST framework
- Mapbox GL JS
- PyTorch 及相关视觉模型
- 通义千问多模态（DashScope）

以及帮助我们一起完善社区地图的所有同学和社区伙伴。