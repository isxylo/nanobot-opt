---
name: midas
description: "Call MIDAS NX (Civil NX / Gen NX) REST API. Use this skill when the user wants to create projects, add nodes/elements, assign loads, run analysis, or query results via MIDAS API."
metadata: {"nanobot":{"emoji":"🏗️"}}
---

# MIDAS NX API Skill

## 配置

调用 MIDAS API 前，先确认以下两个参数（如果用户没有提供，主动询问）：
- **base_url**：如 `https://moa-engineers.midasit.cn:443/civil`（不同国家/地区地址不同）
- **MAPI-Key**：MIDAS NX 软件中生成的 API Key

这两个参数可以让用户存储在 workspace 的 `midas_config.json` 中，避免每次询问：
```json
{
  "base_url": "https://moa-engineers.midasit.cn:443/civil",
  "mapi_key": "your-key-here"
}
```

## API 调用方式

使用 Python + requests 发送请求，**发送前必须用 jsonschema 验证请求体**：

```python
import requests, json

base_url = "https://moa-engineers.midasit.cn:443/civil"  # 从配置读取
mapi_key = "your-key"  # 从配置读取

headers = {
    "Content-Type": "application/json",
    "MAPI-Key": mapi_key
}

# 加载 API 文档
with open('/root/.nanobot/workspace/midas_api.json') as f:
    api_docs = json.load(f)

def validate_body(uri: str, body: dict) -> None:
    """发送前验证请求体是否符合 Schema，不符合直接抛出异常。"""
    import jsonschema
    # 找到对应接口的 schema
    entry = next((v for v in api_docs.values() if v.get('uri') == uri.lstrip('/')), None)
    if not entry or not entry.get('schema'):
        return  # 没有 schema 则跳过验证
    try:
        schema = json.loads(entry['schema']) if isinstance(entry['schema'], str) else entry['schema']
        # schema 顶层是 {"API_NAME": {actual_schema}}，取内层
        if len(schema) == 1:
            schema = list(schema.values())[0]
        jsonschema.validate(instance=body, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"请求体验证失败: {e.message}\n路径: {list(e.path)}")
    except json.JSONDecodeError:
        pass  # schema 解析失败则跳过验证

def midas_api(method, sub_url, body=None):
    # 发送前验证
    if body and method in ("POST", "PUT"):
        validate_body(sub_url, body)
    url = base_url + sub_url
    if method in ("POST", "PUT"):
        r = requests.request(method, url, headers=headers, json=body, timeout=120)
    else:
        r = requests.request(method, url, headers=headers, timeout=120)
    return r.json()
```

## 查询 API 文档

完整的 API 参考存储在 `/root/.nanobot/workspace/midas_api.json`，包含 440 个接口。

查询特定接口：
```python
import json
with open('/root/.nanobot/workspace/midas_api.json') as f:
    apis = json.load(f)

# 按 title 搜索
results = [(k,v) for k,v in apis.items() if '关键词' in v['title'].lower()]
for k,v in results:
    print(v['title'], '|', v['uri'], '|', v['http_method'])
    print('Schema:', v['schema'][:200] if v['schema'] else None)
    print('Example:', v['examples'][0][:200] if v['examples'] else None)
```

## 常用接口速查

| 功能 | URI | Method |
|---|---|---|
| 新建项目 | `/doc/new` | POST |
| 修改单位 | `/db/unit` | PUT |
| 创建节点 | `/db/node` | POST |
| 创建单元 | `/db/elem` | POST |
| 添加荷载 | `/db/bsload` 等 | POST |
| 运行分析 | `/db/anal` | POST |
| 获取结果 | `/post/...` | GET |

## 使用流程

1. **理解用户意图**：用户说「创建节点」「添加荷载」等
2. **查询 midas_api.json**：找到对应接口的 URI、Method、Schema、Examples
3. **构造请求体**：参考 Schema 和 Examples，结合用户提供的参数
4. **用 exec 执行 Python 脚本发送请求**
5. **解析并展示返回结果**

## 示例：新建项目并创建节点

```python
import requests

base_url = "https://moa-engineers.midasit.cn:443/civil"
headers = {"Content-Type": "application/json", "MAPI-Key": "your-key"}

# 1. 新建项目
r = requests.post(base_url + "/doc/new", headers=headers, json={"Argument": {}})
print("新建项目:", r.json())

# 2. 创建节点
body = {
    "Assign": {
        "1": {"X": 0, "Y": 0, "Z": 0},
        "2": {"X": 0, "Y": 0, "Z": 3}
    }
}
r = requests.post(base_url + "/db/node", headers=headers, json=body)
print("创建节点:", r.json())
```

## 注意事项

- MIDAS NX 软件必须在目标电脑上**运行中**，且已开启 API 服务
- base_url 中的 IP/域名是运行 MIDAS NX 的那台电脑，不是服务器
- API Key 在 MIDAS NX 软件的设置中生成
- 所有请求均为同步，分析计算可能耗时较长（建议 timeout 设置 120s 以上）
