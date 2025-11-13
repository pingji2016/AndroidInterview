# Android 开发常见面试问题

## 基础篇

### 1. Activity 生命周期
- 完整生命周期方法调用顺序 [查看说明文档](./sources/1/1.0.md)
- 完整生命周期方法调用顺序 [查看说明文档](./sources/1/1.1.md)
- 横竖屏切换时的生命周期变化 [查看说明文档](./sources/1/1.2.md)
- onSaveInstanceState() 和 onRestoreInstanceState() 的调用时机 [查看说明文档](./sources/1/1.3.md)
- 启动模式 (standard, singleTop, singleTask, singleInstance) 的区别和应用场景 [查看说明文档](./sources/1/1.4.md)

### 2. Fragment 相关
- Fragment 生命周期以及与 Activity 生命周期的关系 [查看说明文档](./sources/2/2.1.md)
- add() 和 replace() 的区别  [查看说明文档](./sources/2/2.2.md)
- show()/hide() 与 detach()/attach() 的区别 [查看说明文档](./sources/2/2.3.md)
- Fragment 之间通信的方式 [查看说明文档](./sources/2/2.4.md)
- Fragment 之间Jepack导航文件 [查看说明文档](./sources/2/2.5.md)

### 3. 组件通信
- Binder 驱动跨进程通信 [查看说明文档](./sources/3/3.0.md)
- Intent 的显式和隐式调用 [查看说明文档](./sources/3/3.1.md)
- BroadcastReceiver 的两种注册方式及区别 [查看说明文档](./sources/3/3.4.md)
- ContentProvider 的实现原理 [查看说明文档](./sources/3/3.2.md)
- 使用 AIDL 进行跨进程通信的基本步骤 [查看说明文档](./sources/3/3.3.md)

## 进阶篇

### 1. 性能优化
- ANR 的产生原因及预防措施 [查看说明文档](./sources/4/4.1.md)
- 布局优化的常用手段  [查看说明文档](./sources/4/4.2.md)
- 图片加载优化 (Bitmap 处理) [查看说明文档](./sources/4/4.3.md)
- ThreadLocal 引发内存泄漏的机制  [查看说明文档](./sources/10/10.md)

### 2. 多线程
- Handler 机制原理 [查看说明文档](./sources/5/5.1.md)
- AsyncTask 的优缺点 [查看说明文档](./sources/5/5.2.md)
- IntentService 的特点 [查看说明文档](./sources/5/5.3.md)
- 线程池的正确使用 [查看说明文档](./sources/5/5.4.md)

### 3. 网络相关
- OkHttp 拦截器原理 [查看说明文档](./sources/other/5_1.1.md)
- Retrofit 动态代理实现原理 [查看说明文档](./sources/other/5_1.2.md)
- HTTP 和 HTTPS 的区别 [查看说明文档](./sources/other/5_1.3.md)
- WebView 优化及安全注意事项 [查看说明文档](./sources/other/5_1.4.md)

## 架构篇


### 1. 设计模式
- MVC、MVP、MVVM 的区别 [查看说明文档](./sources/6/6.1.md)
- 单例模式的正确实现 [查看说明文档](./sources/6/6.2.md)
- 观察者模式在 Android 中的应用 [查看说明文档](./sources/6/6.3.md)
- 依赖注入的优势 [查看说明文档](./sources/6/6.4.md)

### 2. Jetpack 组件
- ViewModel 的生命周期管理  [查看说明文档](./sources/7/7.1.md)
- LiveData 与 RxJava 的区别  [查看说明文档](./sources/7/7.2.md)
- Room 数据库升级方案  [查看说明文档](./sources/7/7.3.md)
- WorkManager 的使用场景  [查看说明文档](./sources/7/7.4.md)

### 3. 模块化/组件化
- ARouter 实现原理 [查看说明文档](./sources/8/8.1.md)
- 组件间通信方案 [查看说明文档](./sources/8/8.2.md)
- 资源冲突解决方案 [查看说明文档](./sources/8/8.3.md)
- 动态加载原理 [查看说明文档](./sources/8/8.4.md)

## AOSP篇
- AMS、PMS、WMS 和 SystemServer 详解 [查看说明文档](./sources/8/8.0.md)
- Android 系统启动流程 [查看说明文档](./sources/6/6.0.md)
- 常见的ADB命令 [查看说明文档](./sources/8/8.5.md)
- fastboot， bl锁，系统分区之间的关系 [查看说明文档](./sources/8/8.6.md)
- android系统目录，预装应用安装在哪，普通应用可以访问那些目录 [查看说明文档](./sources/8/8.7.md)
- ab分区，fastboot刷机，9008刷机的关系 [查看说明文档](./sources/8/8.8.md)

## 实战篇

### 1. 自定义 View
- View 的测量、布局、绘制流程  [查看说明文档](./sources/9/9.1.md)
- 事件分发机制  [查看说明文档](./sources/9/9.2.md)
- 自定义属性实现  [查看说明文档](./sources/9/9.3.md)
- 性能优化注意事项  [查看说明文档](./sources/9/9.4.md)
- RecyclerView的缓存机制 [查看说明文档](./sources/9/9.5.md)
- 高斯模糊实现 [查看说明文档](./sources/9/9.6.md)
- 游戏渲染 [查看说明文档](./sources/9/9.7.md)
- CameraX调用流程 [查看说明文档](./sources/9/9.8.md)

### 2. 动画
- 补间动画和属性动画的区别  [查看说明文档](./sources/10/10.1.md)
- 属性动画原理  [查看说明文档](./sources/10/10.2.md)
- 转场动画实现  [查看说明文档](./sources/10/10.3.md)
- Lottie 动画使用  [查看说明文档](./sources/10/10.4.md)

### 3. 新技术
- Compose 与传统 UI 开发的对比 [查看说明文档](./sources/11/11.1.md)
- Kotlin 协程原理 [查看说明文档](./sources/11/11.2.md)
- Flutter 混合开发方案 [查看说明文档](./sources/11/11.3.md)
- 鸿蒙系统适配考虑 [查看说明文档](./sources/11/11.4.md)

## 逆向

### 1. 新技术

## 疑难问题
1. 如何实现应用保活？[查看说明文档](./sources/12/12.1.md)
2. 如何减少包体积？[查看说明文档](./sources/12/12.2.md)
3. 如何实现多渠道打包？[查看说明文档](./sources/12/12.3.md)
4. 如何处理 64K 方法数限制？[查看说明文档](./sources/12/12.4.md)
5. 如何实现热修复？[查看说明文档](./sources/12/12.5.md)

## 开放性问题

1. 你是如何进行技术选型的？[查看说明文档](./sources/13/13.1.md)
2. 遇到最难解决的 Bug 是什么？如何解决的？[查看说明文档](./sources/13/13.2.md)
3. 如何设计一个图片加载框架？[查看说明文档](./sources/13/13.3.md)
4. 如何实现应用秒开？[查看说明文档](./sources/13/13.4.md)
5. 你如何看待 Flutter 和原生开发的未来？[查看说明文档](./sources/13/13.5.md)

## 学习资料推荐
- google官网[查看连接](https://source.android.com/docs/core/virtualization/architecture?hl=zh-cn)
- Docker使用 [查看说明文档](./sources/14/14.0.md)
  
1. 
准备面试时，建议结合自身项目经验，对每个知识点都能举出实际应用案例，这将大大增加回答的说服力。同时，保持对新技术趋势的关注也很重要。


## Google官方文档
### 1. AOSP
1. 官方文档[查看链接](https://source.android.com/docs/setup/download?hl=zh-cn)
2. 厂商ROM定制[查看链接](./sources/14/14.2.md)


### 2. 开发
1. Core Area[查看链接](https://developer.android.com/develop?hl=zh-cn#core-areas)
2. Android 中的权限 [查看说明文档](./sources/16/16.0.md)


### 3，工具类
1. Android studio 监控 profile [查看说明文档](./sources/17/17.0.md)
2. android的反编译方法 [查看说明文档](./sources/17/17.1.md)
3. android的加固和反调试检测方法 [查看说明文档](./sources/17/17.2.md)

### 4，应用
1. 如何实现应用跟新，热更新和冷更新[查看说明文档](./sources/18/18.0.md)
