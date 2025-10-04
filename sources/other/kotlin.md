# Kotlin 面试的10个常见问题

## 1. Kotlin与Java的主要区别是什么？
- **空安全**：Kotlin在类型系统中内置了空安全
- **扩展函数**：无需继承或装饰器即可扩展类功能
- **数据类**：自动生成`equals()`, `hashCode()`, `toString()`等方法
- **协程**：轻量级线程替代方案
- **智能转换**：自动类型推断和转换
- **默认参数和命名参数**：减少方法重载

## 2. 解释Kotlin中的空安全机制
Kotlin通过类型系统区分可空和非空引用：
- `String` - 非空字符串
- `String?` - 可空字符串
- 使用安全调用操作符`?.`：`user?.address?.street`
- Elvis操作符`?:`提供默认值：`val name = user.name ?: "Unknown"`
- 非空断言`!!`强制转换但可能抛出NPE

## 3. 什么是Kotlin的数据类(data class)？
```kotlin
data class User(val name: String, val age: Int)
```
自动生成：
- `equals()`/`hashCode()`
- `toString()`格式如"User(name=John, age=42)"
- `componentN()`函数用于解构声明
- `copy()`函数

## 4. 解释Kotlin中的扩展函数
允许在不修改类定义的情况下扩展类功能：
```kotlin
fun String.addExclamation() = "$this!"

// 使用
"Hello".addExclamation() // 返回"Hello!"
```
- 编译为静态方法
- 不会实际修改原始类
- 接收者类型决定函数作用域

## 5. Kotlin协程与线程的区别
- **轻量级**：一个线程可运行多个协程
- **挂起而非阻塞**：挂起时释放线程资源
- **结构化并发**：通过作用域管理生命周期
- **更简单的异步代码**：顺序编写异步逻辑
- **取消支持**：内置取消机制

## 6. Kotlin中的`lateinit`和`by lazy`的区别
- **`lateinit`**：
  - 用于var变量
  - 必须是非空类型
  - 不能用于原始类型(Int, Boolean等)
  - 可重新赋值

- **`by lazy`**：
  - 用于val常量
  - 延迟初始化，首次访问时计算
  - 线程安全选项可用
  - 初始化后不可变

## 7. Kotlin中的密封类(sealed class)是什么？
```kotlin
sealed class Result {
    data class Success(val data: String) : Result()
    data class Error(val message: String) : Result()
}
```
特点：
- 限制继承层级
- 所有子类必须在同一文件中声明
- 与when表达式配合使用时，编译器能检查是否覆盖所有情况

## 8. 解释Kotlin中的委托属性
```kotlin
var token: String by Delegates.observable("<none>") { 
    prop, old, new -> println("Token changed from $old to $new")
}
```
常见委托：
- `lazy`：延迟初始化
- `observable`：属性变化通知
- `vetoable`：允许否决属性变更
- 自定义委托实现`getValue`/`setValue`

## 9. Kotlin中的内联函数(inline)有什么作用？
```kotlin
inline fun <T> lock(lock: Lock, body: () -> T): T {
    lock.lock()
    try {
        return body()
    } finally {
        lock.unlock()
    }
}
```
优点：
- 减少高阶函数的运行时开销
- 允许非局部返回(从lambda中返回外层函数)
- 可配合`reified`类型参数实现类型擦除时的类型访问

## 10. Kotlin与Java的互操作性如何实现？
- Kotlin可100%调用Java代码
- Java也可调用大部分Kotlin代码
- 互操作特性：
  - 空安全注解(`@Nullable`, `@NotNull`)
  - 属性访问(Java字段getter/setter映射为Kotlin属性)
  - 单例对象(`object`类编译为静态字段)
  - 扩展函数编译为静态方法
  - 伴生对象成员作为静态成员
  - Jvm注解控制字节码生成(`@JvmStatic`, `@JvmOverloads`等)

这些问题涵盖了Kotlin的核心概念和特性，适合评估候选人对Kotlin的掌握程度。