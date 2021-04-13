# PTTS 网页演示

[TOC]

使用 [Flask](https://github.com/pallets/flask) + [Vue](https://github.com/vuejs/vue)（框架：[Vuetify](https://github.com/vuetifyjs/vuetify)）完成的语音合成单网页演示项目，语音合成后端基于我的另一个项目 [atomicoo/ParallelTTS](https://github.com/atomicoo/ParallelTTS)。

## 目录结构

```
.
|--- backend/
     |--- pretrained/  # 预训练模型
     |--- mytts.py     # 封装 TTS 类
     |--- ...
|--- dist/             # 前端的编译输出
|--- frontend/
     |--- public/
     |--- src/
          |--- components/
               |--- MyParaTTS.vue  # 语音合成页面
          |--- ...
     |--- ...
|--- client.py         # 接口测试脚本
|--- LICENSE
|--- README.md         # 说明文档
|--- requirements.txt  # 依赖文件
|--- server.py         # 服务器端启动脚本
```

## 快速开始

```shell
$ git clone https://github.com/atomicoo/PTTS-WebAPP.git
$ cd PTTS-WebAPP/frontend/
$ npm install --save
$ npm run dev
$ cd ..
$ pip install -r requirements.txt
$ python server.py
$ python client.py
```

运行 `npm run dev` 命令后，项目根目录下应该已经生成前端代码的编译输出，在 `./dist/` 目录下。

运行 `python server.py` 命令后，服务器端已经启动，可以先试试 `python client.py` 测试一下语音合成接口是否正常。

如果至此一切正常，那直接访问 http://localhost:5000/ 即可。

![image-20210412175503742](https://cdn.jsdelivr.net/gh/atomicoo/picture-bed@latest/2021/04/image-20210412175503742.png)

## 一些问题

- 语音合成后端基于我自己的另一个项目 [atomicoo/ParallelTTS](https://github.com/atomicoo/ParallelTTS)，但为了简化重构了代码结构，如果想要换成其他语言的话，理论上只需要替换掉 `./config/` 下的配置文件和 `./pretrained/` 下的模型文件即可，但没有经过完全测试，不能确保不会出现问题。
- ~~目前只支持调整语速，后续会增加音量和语调的调整，如果有大佬能帮忙搞定就更好了 [doge]。~~（已完成）
- 语调的调整使用 <u>变速不变调（TSM）+ 重采样</u> 方案来完成；音量的调整使用比较简单粗暴的方式，后续会改掉。

## 参考资料

- [Flask：Python Web 微框架](https://flask.palletsprojects.com/en/1.1.x/)
- [Vuetify：Material Design 框架](https://vuetifyjs.com/zh-Hans/)
- [变速不变调方法总结 - 知乎](https://zhuanlan.zhihu.com/p/337193578)