

## 框架结构

为了实现linux系统下的高性能并发，我们还需要一个高性能的框架

同时需保证服务器安装有nginx、gunicorn、并安装gevent库，nginx用于配置负载均衡，gunicorn+gevent保证启用python多线程和正确的工作模式

![image-20210729124915777](https://i.loli.net/2021/07/29/j1XDf4JinMPGHs5.png)

flask是一个比较轻的框架，当然很多功能都要自己实现。用flask自带的webserver，每收到一个请求会新建一个线程，所以请求并发数高的话会很慢。gunicorn来实现webserver，会启动多个进程，通过gevent模式来运行，会达到协程的效果，每来一个请求会自动分配给不同的进程来执行

### 项目结构

#### app

没什么好说的，django、tornado都有这个文件夹，整个项目的逻辑代码都在这里

#### app.__ init__

在这里可以写入项目启动时要加载的内容，以及注册蓝本。额外初始化的内容可以根据不同的环境进行不同的初始化操作

#### app.base

描述接收请求时的操作，对请求数据进行校验，并且可以针对不同的场景有两种实现，BaseResource是restful的标准写法，但只能接收json，自编的MyBaseResource可以实现json和text兼容

#### app.firstApi

api的视图层代码，继承BaseResource后，init方法内可以通过 self.parser.add_argument()定义请求体字段

#### app._apis

完成一个api的view层代码后，在__init__.py中注册新的api

#### app.common

公共功能放在这里，比如日志等

#### app.utils

一些工具类，其实和common差不多

#### app.views

这里用于注册蓝本，在项目启动时会执行

#### app.extensions

项目启动时初始化一些额外的操作

#### conf

配置文件。其中config.py可以写一个基类，和若干个继承基类的子类，来实现不同环境时加载不同的配置

#### gun_log

项目启动时，该文件夹下会生成一个内容是该项目的进程ID的文件，方便kill

#### gun.py

配置要监听的端口号，以及gun本身的日志的配置

#### manage.py

项目启动时，首先加载gun.py，然后就加载manage，这里创建app对象



## 运行

也可以设置访问远端服务器，如阿里云`121.199.66.66`

### 服务器运行

![image-20210730151913520](https://i.loli.net/2021/07/30/9sEbljFGwRH65cv.png)

启动命令`gunicorn -c gun.py manage:app`：会要求以gevent的工作模式启动，并自动配置启用进程数



## 服务：关键词提取

#### 接口介绍

关键词提取函数`keyword_extract()`提供了从文本中提取关键词的功能。

```python
keyword_extract(text: str, *method: int) -> list
```

##### 输入

* **text**: 接受字符串，可以是文本内容，也可以是文件路径。

* **method**: 为可选项，无参数传入时使用jieba库提供的方法；传入参数与实际使用的方法对应关系为：

| method=?   | 实际使用方法                   |
| ---------- | ------------------------------ |
| 0(default) | `jieba.analyse.extract_tags()` |
| 1          | LDA                            |
| 2          | RaKUn                          |
| 3          | TF-IDF                         |

##### 输出

关键词组，以**list**类型存放，例如：

```python
{'纳税人', '企业', "增值税"}
```

**完整代码**

```python
_get_abs_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), path))
_get_module_path = lambda path: os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), path))


def keyword_extract(text: str, *method: int):
    if os.path.isfile(_get_abs_path(text)):
        f = open(_get_abs_path(text), 'rb')
        text = str(f.read(), encoding='utf-8')

    data = {
        "text":text,
        "method":method
    }

    url = 'http://121.199.66.66:5000/kwe'
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url=url, headers=headers, data=json.dumps(data))
    res = json.loads(response.text).get('keyword')
    return res
```

通过绝对路径判断文件是否存在并正确。传入如果是普通字符串，直接封装进json，如果是文件，将其读取并以utf-8编码后封装

将方法选择参数直接转发

阿里云服务器公网地址`http://121.199.66.66`，在5000端口下调用kwe服务

```python
@app.route('/kwe',methods=['POST'])
def keyword_extract():
    print("方法进入")

    res = []
    my_json = request.get_json()
    text = my_json.get("text")
    method = my_json.get("method")

    if len(method) == 0:
        res = jieba_method(text)

    if len(method) == 1:
        if method[0] == 0:
            # jieba method
            res = jieba_method(text)
        elif method[0] == 1:
            # LDA
            res = lda.keyword_extraction(text)
        elif method[0] == 2:
            # RaKUn method
            res = ra.rakun(text, 'file')
        elif method[0] == 3:
            # TF-IDF
            res = ti.tfidf_extract(text)
        else:
            print('2nd param should be 0~3')
    else:
        print('there should be only 2 params')

    print(res)
    return jsonify({'keyword': res})
```

app的视图函数中，根据方法继续将任务分发，0/1/2/3分别指定jieba/lda/rakun/tfidf方法，将各方法返回的list封装成json，再推送回前端

front将json解开成list，返回给接口



#### 算法适用性

整合了四种方法，适用于不同的环境。其中：

* jieba，作为默认情况，广为大家所熟知的第三方库，泛用性好
* LDA主题模型，在处理大批量内容文件时速度显著，准确性有保障
* RaKUn算法，作为图算法对语义逻辑考虑更周全，适用于文本输入或小型文件输入
* TF-IDF算法，基于统计的经典方法，泛用性好



#### 展示

**网络接口测试**

![00642d9f9a5728b0096b4b5e246ef28](https://i.loli.net/2021/07/29/taSomyxLWvwPDTU.png)



**通过标准接口调用**

![7c2dfe4a63857e5ae9beb3ec3859278](https://i.loli.net/2021/07/29/83LKmNfMXidY5R7.png)



**服务器端效果**

![image-20210729124727183](https://i.loli.net/2021/07/29/1MeRwPUlScV6pJy.png)




