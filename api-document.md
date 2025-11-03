# 社区地图后端接口文档

本文档依据当前代码（`index.html` 前端请求、`map/urls.py`、`myapp/urls.py`、`myapp/views.py`、`myapp/serializers.py`、`myapp/models.py`）整理。

## 说明
- 后端基地址示例：`http://47.109.150.241:8000`
- 统一使用 JSON 作为请求与响应体，除文件上传外。
- 发生错误时，若为 DRF 默认错误返回，通常为字段错误字典；部分自定义接口返回 `{ "error": string }`。

---

## 路由概览
- `map/urls.py` 挂载：
  - `'' -> include('myapp.urls')`
  - `'api/' -> include('myapp.urls')`
  - 另外直挂：
    - `locations/ -> LocationListCreateView`
    - `api/favorites/ -> UserFavoritesView`
    - `api/recognize/ -> LandmarkRecognitionView`
    - `hello/ -> hello_view`
- `myapp/urls.py`：
  - `locations/ -> LocationListCreateView`（与下条重复路径定义，见说明）
  - `locations/<int:pk>/ -> LocationDetailView`
  - `upload-image/ -> ImageUploadView`
  - `locations/ -> LocationListAPI`（与上方重复；实际生效取决于装载顺序，见说明）

说明：`myapp/urls.py` 中存在两条同为 `locations/` 的路由（分别指向 `LocationListCreateView` 与 `LocationListAPI`）。在 Django 中后定义的会覆盖前定义的匹配。当前文件顺序显示 `LocationListAPI` 在最后，可能会覆盖 `LocationListCreateView` 的列表创建功能，导致实际响应为 `LocationListAPI`。同时在 `map/urls.py` 又直挂了 `locations/ -> LocationListCreateView`，因此不同入口访问时可能得到不同实现。前端请求使用了 `GET /api/locations`，由于 `map/urls.py` 将 `'api/'` 下也 include 了 `myapp.urls`，最终会命中 `myapp.urls` 中的 `locations/`。请注意该重复定义带来的不一致风险。

---

## 实体模型概览
- `Location`：
  - 字段：`id`, `name`, `latitude`, `longitude`, `description`, `image`, `owner`, `favorites`
  - `favorites`：多对多到 `User`，related_name=`favorite_locations`
- 主要序列化：`LocationGeoJSONSerializer`，返回 GeoJSON 风格：
  ```json
  {
    "type": "Feature",
    "properties": {
      "id": 1,
      "name": "...",
      "description": "...",
      "image": "<绝对或完整URL>",
      "owner": "<owner username>"
    },
    "geometry": {
      "type": "Point",
      "coordinates": [ <longitude>, <latitude> ]
    }
  }
  ```

---

## 接口清单

### 1) GET /api/locations
- 功能：获取所有地标（Location）列表。
- 前端调用：`index.html` 的 `loadLandmarks()` 使用 `fetch(`${API_BASE_URL}/api/locations`)`。
- 可能命中视图：`LocationListAPI.get` 或 `LocationListCreateView.get`（取决于路由覆盖，见路由概览）。
- 期望/实际返回：
  - 代码实际返回为 `LocationGeoJSONSerializer` 的数组（无统一的 `success`/`data` 包裹）：
    ```json
    [
      {
        "type": "Feature",
        "properties": { ... },
        "geometry": { ... }
      }
    ]
    ```
- 备注：前端当前期望 `{ success: boolean, data: Feature[] }` 结构（`index.html` 第 556 行判断 `result.success`）。与后端现状不一致。建议后端适配或前端调整解析逻辑。

### 2) POST /api/locations
- 功能：创建地标。
- 可能命中视图：`LocationListAPI.post` 或 `LocationListCreateView.post`。
- 请求体（JSON）：
  ```json
  {
    "name": "string",
    "description": "string",
    "category": "string",  // 注意：模型中无 category 字段，若传入将被忽略或报错
    "latitude": 30.0,
    "longitude": 104.0
  }
  ```
- 返回：
  - 若命中 `LocationListAPI.post` 且校验通过，返回创建后的单个 `LocationGeoJSONSerializer` 对象（无 `success` 包裹），HTTP 201。
  - 校验失败返回字段错误字典，HTTP 400。
- 鉴权：`LocationListCreateView` 要求 `IsAuthenticatedOrReadOnly`，`LocationListAPI` 的 post 未显式声明权限，默认允许。路由覆盖不同会导致权限与行为差异。

### 3) GET /api/locations/<id>
- 功能：获取单个地标详情。
- 视图：`LocationDetailView.get`。
- 返回：单个 `LocationGeoJSONSerializer` 对象。
- 鉴权：`IsAuthenticatedOrReadOnly`（读取允许匿名）。

### 4) PUT/PATCH /api/locations/<id>
- 功能：更新地标。
- 视图：`LocationDetailView.put/patch`。
- 请求体：与 `Location` 字段对应（GeoJSON Serializer 接收普通字段，注意其自定义 `to_representation` 仅影响输出）。
- 返回：更新后的单个 `LocationGeoJSONSerializer` 对象。
- 鉴权：`IsAuthenticatedOrReadOnly`。

### 5) DELETE /api/locations/<id>
- 功能：删除地标。
- 视图：`LocationDetailView.delete`。
- 返回：DRF 默认删除响应（通常为空或 `{}`，HTTP 204）。
- 前端调用：`index.html` 的 `deleteLandmark(id)` 使用 `DELETE ${API_BASE_URL}/landmarks/${id}`，与后端路径不一致（后端为 `/api/locations/<id>`）。建议修正前端或增加兼容路由。

### 6) GET /api/favorites/
- 功能：获取当前用户收藏的地标列表。
- 视图：`UserFavoritesView.get`（在 `map/urls.py` 中定义）。
- 认证：从 Cookie 读取 `user_id`，否则 401。
- 返回：`LocationGeoJSONSerializer` 数组。
  ```json
  [ { "type": "Feature", ... } ]
  ```
- 错误：未找到用户返回 `{ "error": "User not found" }`，HTTP 404。

### 7) POST /api/recognize/
- 功能：地标识别（上传图片）。
- 视图：`LandmarkRecognitionView.post`。
- 请求：`multipart/form-data`，字段 `image` 为文件。
- 成功返回：
  ```json
  {
    "landmark": "string",
    "confidence": 0.95,
    "image_url": "/media/temp/<file>"
  }
  ```
- 失败返回：
  - 缺少文件：`{ "error": "未提供图片" }`，HTTP 400
  - 服务异常：`{ "error": "..." }`，HTTP 500

### 8) POST /api/upload-image/
- 功能：上传图片并进行识别（另外一条上传接口）。
- 视图：`ImageUploadView.post`（文件内有两次定义，后者生效）。
- 路由：`myapp/urls.py` 中为 `upload-image/`，因此完整路径通常为 `/api/upload-image/` 与 `/upload-image/`（取决于 include 的前缀）。
- 请求：`multipart/form-data`，字段 `image` 为文件。
- 返回：
  ```json
  {
    "image_url": "/media/location_images/<file>",
    "recognition": <recognize_image(save_path) 的返回结构>
  }
  ```

### 9) GET /hello/
- 功能：健康检查/示例。
- 返回：纯文本 `"hello"`。

---

## 前端与后端路径/结构不一致点
- 前端：
  - 读取列表：调用 `GET /api/locations`，并期望 `{ success, data }` 结构；后端当前直接返回数组。
  - 创建：调用 `POST /landmarks`（而非 `/api/locations`），且发送 `category` 字段；后端 `Location` 模型没有 `category` 字段。
  - 删除：调用 `DELETE /landmarks/{id}`（而非 `/api/locations/{id}`）。
  - 同步：额外向 `POST /map/location` 发送 GeoJSON Feature 数组；后端未实现该路由。
- 建议：
  1. 统一后端接口为 `/api/locations` 风格，并在响应中加入 `{ success, data }`；或修改前端按当前后端返回结构解析。
  2. 为 `landmarks` 路径增加后端兼容别名（可在 `map/urls.py` 增加 `path('landmarks/', ...)` 和 `path('landmarks/<int:pk>/', ...)`）。
  3. 去除或实现 `category` 字段：若需要分类，请在 `Location` 模型中新增 `category` 字段并迁移；否则前端移除该字段。
  4. 删除或实现 `POST /map/location` 的同步接口。
  5. 清理 `myapp/urls.py` 中重复的 `locations/` 路由，避免覆盖导致行为不一致。

---

## 示例请求/响应

- 获取列表：
  Request:
  GET `/api/locations`

  Response 200:
  ```json
  [
    {
      "type": "Feature",
      "properties": {
        "id": 1,
        "name": "示例地标",
        "description": "...",
        "image": null,
        "owner": "admin"
      },
      "geometry": {
        "type": "Point",
        "coordinates": [104.0, 30.0]
      }
    }
  ]
  ```

- 创建：
  Request:
  POST `/api/locations`
  ```json
  { "name": "A", "description": "B", "latitude": 30.0, "longitude": 104.0 }
  ```

  Response 201:
  ```json
  {
    "type": "Feature",
    "properties": { "id": 2, "name": "A", "description": "B", "image": null, "owner": "admin" },
    "geometry": { "type": "Point", "coordinates": [104.0, 30.0] }
  }
  ```

- 删除：
  Request:
  DELETE `/api/locations/2`

  Response 204:
  （空响应或 `{}`）
