# Android 开发常见面试问题

## 基础篇

### 1. Activity 生命周期
- 完整生命周期方法调用顺序 [查看说明文档](./1.1.md)
- 横竖屏切换时的生命周期变化 [查看说明文档](./1.1.md)
- onSaveInstanceState() 和 onRestoreInstanceState() 的调用时机
- 启动模式 (standard, singleTop, singleTask, singleInstance) 的区别和应用场景

### 2. Fragment 相关
- Fragment 生命周期以及与 Activity 生命周期的关系
- add() 和 replace() 的区别
- show()/hide() 与 detach()/attach() 的区别
- Fragment 之间通信的方式

### 3. 组件通信
- Intent 的显式和隐式调用
- BroadcastReceiver 的两种注册方式及区别
- ContentProvider 的实现原理
- 使用 AIDL 进行跨进程通信的基本步骤

## 进阶篇

### 1. 性能优化

- ANR 的产生原因及预防措施
- 布局优化的常用手段
- 图片加载优化 (Bitmap 处理)

### 2. 多线程
- Handler 机制原理
- AsyncTask 的优缺点
- IntentService 的特点
- 线程池的正确使用

### 3. 网络相关
- OkHttp 拦截器原理
- Retrofit 动态代理实现原理
- HTTP 和 HTTPS 的区别
- WebView 优化及安全注意事项

## 架构篇

### 1. 设计模式
- MVC、MVP、MVVM 的区别
- 单例模式的正确实现
- 观察者模式在 Android 中的应用
- 依赖注入的优势

### 2. Jetpack 组件
- ViewModel 的生命周期管理
- LiveData 与 RxJava 的区别
- Room 数据库升级方案
- WorkManager 的使用场景

### 3. 模块化/组件化
- ARouter 实现原理
- 组件间通信方案
- 资源冲突解决方案
- 动态加载原理

## 实战篇

### 1. 自定义 View
- View 的测量、布局、绘制流程
- 事件分发机制
- 自定义属性实现
- 性能优化注意事项

### 2. 动画
- 补间动画和属性动画的区别
- 属性动画原理
- 转场动画实现
- Lottie 动画使用

### 3. 新技术
- Compose 与传统 UI 开发的对比
- Kotlin 协程原理
- Flutter 混合开发方案
- 鸿蒙系统适配考虑

## 疑难问题

1. 如何实现应用保活？
2. 如何减少包体积？
3. 如何实现多渠道打包？
4. 如何处理 64K 方法数限制？
5. 如何实现热修复？

## 开放性问题

1. 你是如何进行技术选型的？
2. 遇到最难解决的 Bug 是什么？如何解决的？
3. 如何设计一个图片加载框架？
4. 如何实现应用秒开？
5. 你如何看待 Flutter 和原生开发的未来？

准备面试时，建议结合自身项目经验，对每个知识点都能举出实际应用案例，这将大大增加回答的说服力。同时，保持对新技术趋势的关注也很重要。