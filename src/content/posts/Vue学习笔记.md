---
title: Vue学习随笔
published: 2025-09-20
description: Vue入门一篇通.
draft: false
---

前言：本博客是ky的Vue学习笔记，个人观点，若有错误，欢迎指正。有些内容可能没有那么详细，需要你借助Ai或其他资料自行了解

# 一、Vue工程创建

## 1、认识Vue工具

在Vue学习中，最先让我困惑的就是各种Vue的工具链，Vue的工具链主要有几个

- Vue的构建工具链：Vue官方脚手架Vue CLI,先进的Vite工具链
- Vue的状态管理工具：Vuex用于管理应用中各个组件之间的状态
- Vue的路由管理工具：Vue Router用于实现不同页面之间的跳转和状态管理、
- Vue的组件库：如Element UI用于实现更美的界面，不用自己手搓样式表

## 2、构建第一个Vue程序

现在我在Idea中使用Vite来构建我的第一个Vue项目

- 首先在新建项目中选择vite,并且更改模板为Vue后点击创建
- 现在就可以使用vite工具链安装相关依赖` npm install packagename`或者` yarn add packagename`
- 在编写完程序以后，我们需要编译运行整个项目` npm run dev`或者直接点击运行（需要将运行调试修改为npm并选择正确的脚本），此时在终端会返回一个本地端口地址，点开就可以看到Vue的初始网页了

## 3、项目结构

接着我们来说一下Vue项目的项目结构:

```my-vue-project/          # 项目根目录 
my-vue-project/          # 项目根目录 
├─ node_modules/         # 依赖包（npm/yarn 生成，无需手动改） 
├─ public/               # 静态资源（如 favicon、不参与构建的文件） 
│  └─ favicon.ico        # 网站图标示例 
├─ src/                  # 源码核心目录 
│  ├─ assets/            # 资源（图片、样式等） 
│  │  └─ logo.png        # 示例图片 
│  ├─ components/        # 公共组件（可复用的 Vue 组件） 
│  │  └─ HelloWorld.vue  # 示例组件 
│  ├─ App.vue            # 根组件（应用入口组件）
│  ├─ main.js            # 项目入口文件（初始化 Vue 应用） 
│  └─ style.css          # 全局样式 
├─ .gitignore            # Git 忽略规则 
├─ index.html            # 页面模板（Vue 挂载的载体） 
├─ package.json          # 项目配置（依赖、脚本命令等） 
├─ package-lock.json     # npm 依赖版本锁（确保依赖一致） 
├─ vite.config.js        # Vite 配置文件（自定义构建、开发服务器等） 
└─ README.md             # 项目说明文档
```

在Vue中，渲染到网页的界面是index.html文件，它包含了你所编写的所有的组件components，每个组件可以像html标签一样通过<ComponentName/>直接渲染到网页上，这就是组件化设计



# 二、组件文件分析

## 1、文件结构

- 首先是最上方的<script setup></script>（注：setup是新版Vue的语法糖，后面就无需如将数据、方法、计算属性等分散在data、methods、computed）这里是编写响应式API的地方,可以实现动态渲染变量等
- 在中间的是<template></template>这里是组件渲染的主要内容，也是界面呈现的主要内容
- 最下方是<style scoped></style>这里储存的是该组件的样式表，其中scoped是将样式限制在该组件内，防止同名样式之间的污染

## 2、组件导出及杂谈

在组件编写完成之后需要将组件导出，即暴露给其他组件，如果使用上文的<script setup></script>的话，有了setup就不需要自己手动导出，会自动分配文件同名的组件名。但如果你不采用setup语法糖，则需要通过在<script>中export default来导出

在杂谈这里我想谈谈Vue中的命名方式，在组件或文件命名时采用大驼峰命名，props和变量方法名可以采用小驼峰命名，事件以及css可以采用kebab-case命名



# 三、基本指令

来说一下一些Vue中的基本指令

## 1、v-bind

可以简写为:=,仅仅只是实现单项绑定，数据变化会影响视图，视图变化不会影响数据,即在网页页面修改图片地址，并不会改变图片的真实地址

```vue
<script setup>
import { ref } from 'vue'
const imgUrl = ref('https://picsum.photos/200/300')
const isDisabled = ref(true)
</script>

<template>
  <img :src="imgUrl" alt="示例图片">
  <button :disabled="isDisabled">禁用按钮</button>
</template>
```

## 2、v-model

用来绑定用户输入，传递到某个变量，以便于逻辑处理，如：

```vue
<script setup>
  const msg=''
</script>

<template>
  <input type="text" v-model="msg"/>
</template>
```

这个时候用户在输入框输入的东西就会自动储存在msg中

## 3、v-if

顾名思义一个if判断，如果后面的返回值为true,则显示，反之不显示。注意：这里的不显示意味着的是完全销毁，而不是隐藏，当需要时则会重新创建,频繁地创建销毁会导致性能消耗

```vue
<script setup>
let isshow=true
</script>

<template>
  <text v-if="isshow"/>
</template>
```

## 4、v-show

这个和v-if有点像，也是根据后方的语句来判断是否显示，但是与v-if不同的是这里只是隐藏

``````vue
<script setup>
let isshow=true
</script>

<template>
  <text v-show="isshow"/>
</template>
``````

## 5、v-on

完整的格式为v-on:事件名=“表达式或方法”，可以简写为@事件，用于绑定事件与函数

``` vue
<script setup>
import { ref } from 'vue'
let text=ref("内容")
function change(){
  text.value="内容1"
}
</script>

<template>
  <text>{{text}}</text>
  <button @click="change"/>
</template>```
```



# 四、compositions API

## 1、响应式API

### ref

首先来说一个比较重要的，ref,这是用来实现数据的动态渲染的，何为动态渲染，即当数据发生改变之时，视图也会因此发生变化。简而言之：如果没有ref,当你改变了一个数据的值之后，他的视图不会随之改变

注意，ref实际上是将他括号中的内容作为value进行打包，你可以理解为打包成了一个对象，所以在改变text的值的时候应该是改变text中的value，而不是text

```vue
<script setup>
import { ref } from 'vue'
let text=ref("内容")
function change(){
  text.value="内容1"
}
</script>

<template>
  <text>{{text}}</text>
  <button @click="change"/>
</template>
```

### reactive

主要用于将普通对象转换为响应式对象，它返回的就是一个响应式的对象本身，不需要额外的 value属性

``` vue
<script setup>
import { reactive } from 'vue'
const state = reactive({
  message: 'Hello Vue',
  list: [1, 2, 3]
})
const changeMessage = () => {
  state.message = 'Changed message'
}
const addToList = () => {
  state.list.push(4)
}
</script>

<template>
  <div>
    <p>{{ state.message }}</p>
    <button @click="changeMessage">改变消息</button>
    <p>{{ state.list }}</p>
    <button @click="addToList">向列表添加元素</button>
  </div>
</template> 
```

简单点来说，在处理基本数据类型的时候通常选择ref,复杂的对象类型的时候一般选择reactive

### computed

正如他的名字一样，这是个响应式的计算函数，即你在调用时就可直接计算出结果,相对于使用函数计算，computed只有在计算对象真正变化的时候才会重新调用，相当于一个有缓存的计算器，但是参数相对于函数来讲更加固定

``` vue
<template>
  <div>
    <p>加数 A: {{ numA }}</p>
    <input v-model="numA" type="number">
    <p>加数 B: {{ numB }}</p>
    <input v-model="numB" type="number">
    <p>两数之和: {{ sum }}</p>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue';

const numA = ref(0);
const numB = ref(0);

const sum = computed(() => {
  return numA.value + numB.value;
});
</script>
```

### watch

相当于一个监视器，当变量变化时就会触发相应的函数,如果监视一个对象的话就需要深度监视` watch(count,(newval,oldval)=>{
  console.log(count.value)},{deep:true})`

``` vue
<script setup>
import {ref, watch} from "vue";
let count = ref(0);
const plus = ()=>{
  count.value++;
}
watch(count,(newval,oldval)=>{
  console.log(count.value)
})
</script>

<template>
  <button @click="plus">{{count}}</button>
</template>
```



## 2、生命周期

这里主要用到的是生命周期的钩子，所以我们介绍一下**Vue3**中的生命周期的钩子

生命周期的钩子写在哪个组件下，就和哪个组件绑定

|       钩子        |                       说明                        |
| :---------------: | :-----------------------------------------------: |
|  onBeforeMount()  |                 挂载开始之前执行                  |
|    onMounted()    |                 挂载完成之后执行                  |
| onBeforeUpdate()  | 数据更新时执行，但在虚拟 DOM 重新渲染和打补丁之前 |
|    onUpdated()    | 数据更新导致的虚拟 DOM 重新渲染和打补丁完成后执行 |
| onBeforeUnmount() |               卸载组件实例之前执行                |
|   onUnmounted()   |              卸载组件实例完成后执行               |

``` vue
<template>
  <div>生命周期演示</div>
</template>

<script setup>
import { 
  onBeforeMount, 
  onMounted, 
  onBeforeUpdate, 
  onUpdated, 
  onBeforeUnmount, 
  onUnmounted 
} from 'vue'

// 1. 挂载前
onBeforeMount(() => {
  console.log('组件即将挂载（DOM 未生成）')
})

// 2. 挂载后
onMounted(() => {
  console.log('组件已挂载（可操作 DOM）')
  // 示例：绑定窗口大小变化事件
  window.addEventListener('resize', handleResize)
})

// 3. 更新前
onBeforeUpdate(() => {
  console.log('数据更新，DOM 即将重新渲染')
})

// 4. 更新后
onUpdated(() => {
  console.log('DOM 已重新渲染')
})

// 5. 卸载前
onBeforeUnmount(() => {
  console.log('组件即将卸载，清理资源')
  // 示例：移除事件监听
  window.removeEventListener('resize', handleResize)
})

// 6. 卸载后
onUnmounted(() => {
  console.log('组件已卸载')
})

// 辅助函数
const handleResize = () => {
  console.log('窗口大小变化')
}
</script>
```

# 五、状态管理

这里我们介绍一个状态管理工具：Vuex

首先讲讲为什么需要Vuex，在小型项目中，组件间的通信可以通过父子间的传参和事件总线来解决，但是当项目变得复杂，组件变多的时候这种传参方式会变得不理想（混乱、复杂）

Vuex的出现就是为了解决这个问题

## 1、安装Vuex

关于如何使用Vuex,首先就是安装``` npm install vuex```,我们可以直接通过npm安装，一般在src文件夹下创建store文件夹用以存储Vuex实例

## 2、Vuex的核心组成

Vuex作为一个状态仓库，储存着单一模块的所有共享状态，一般全局只有一个Vuex实例

### State

是Vuex中用于存储共享状态的地方，所有的共享状态都存储在State之中

``` vue
import { createStore } from 'vuex'

const store = createStore({
  state() {
    return {
      count: 0, 
      userInfo: null
    }
  }
})
```

而对于Vuex中数据的访问，直接通过$store.state.value进行访问,或者使用mapstate映射到组件computed

``` vue
//方法一
<template>  
	<div>当前计数：{{ $store.state.count }}</div>
</template>
```

``` vue
//方法二
<script>
import { mapState } from 'vuex'

export default {
  computed: {
    // 映射 this.count 为 this.$store.state.count
    ...mapState(['count']),
    // 自定义键名：映射 this.user 为 this.$store.state.userInfo
    ...mapState({ user: 'userInfo' })
  }
</script>
```

### Getter

类似于computed,用于从State中派生出新状态，只有当依赖的State发生变化是才会重新计算，否则返回缓存值

``` vue
const store = createStore({
  state() {
    return {
      todos: [
        { id: 1, text: '学习 Vuex', done: true },
        { id: 2, text: '写项目', done: false }
      ]
    }
  },
  getters: {
    // 派生状态：已完成的任务列表
    doneTodos(state) {
      return state.todos.filter(todo => todo.done)
    }
  }
})
```

### Mutation

是改变状态的唯一途径，他有一条核心原则：只支持同步操作，不支持异步执行

``` vue
const store = createStore({
  state() {
    return { count: 0 }
  },
  mutations: {
    // 无载荷的 mutation
    increment(state) {
      state.count++
    },
    // 有载荷的 mutation（载荷通常是对象，可传递多个参数）
    incrementBy(state, payload) {
      state.count += payload.amount
    }
  }
})
```

然后再来个触发Mutation的示例

``` vue
// 1. 触发无载荷的 mutation
store.commit('increment')

// 2. 触发有载荷的 mutation（两种写法）
// 方式 A：直接传值
store.commit('incrementBy', 5)
// 方式 B：对象形式（推荐，更清晰）
store.commit('incrementBy', { amount: 5 })

// 3. 另一种对象式提交（type 指定 mutation 名称）
store.commit({
  type: 'incrementBy',
  amount: 5
})
```

### Action

是用于处理异步操作的（接口请求，定时器），最后通过Mutation来修改State,示例如下

``` Vue
import axios from 'axios'

const store = createStore({
  state() {
    return { userInfo: null, loading: false }
  },
  mutations: {
    setUserInfo(state, userInfo) {
      state.userInfo = userInfo
    },
    setLoading(state, status) {
      state.loading = status
    }
  },
  actions: {
    // 异步获取用户信息
    async fetchUserInfo(context, userId) {
      // 1. 开启加载状态
      context.commit('setLoading', true)
      try {
        // 2. 异步请求（接口调用）
        const res = await axios.get(`/api/user/${userId}`)
        // 3. 触发 mutation 修改状态
        context.commit('setUserInfo', res.data)
      } catch (err) {
        console.error('获取用户信息失败：', err)
      } finally {
        // 4. 关闭加载状态
        context.commit('setLoading', false)
      }
    }
  }
})
```

而对于Action的触发则是通过dispatch来实现的

```   vue
// 触发 action（支持传参和异步等待）
store.dispatch('fetchUserInfo', 123).then(() => {
  console.log('用户信息获取完成')
})

async function getInfo() {
  await store.dispatch('fetchUserInfo', 123)
  console.log('用户信息获取完成')
}
```

### Moudule

是用于模块化编写的，当我们的项目过于庞大之时我们可以将Vue Store拆分为多个Module(模块)，每个模块拥有自己的 `state`、`getters`、`mutations`、`actions`，甚至嵌套子模块。

``` vue
// 模块 A：用户相关状态
const userModule = {
  // 开启命名空间（避免模块间命名冲突）
  namespaced: true,
  state() {
    return { name: '张三', age: 20 }
  },
  mutations: {
    updateName(state, newName) {
      state.name = newName
    }
  }
}



// 模块 B：商品相关状态
const productModule = {
  namespaced: true,
  state() {
    return { list: ['商品 1', '商品 2'] }
  }
}



// 组装 store
const store = createStore({
  modules: {
    user: userModule, // 挂载用户模块，命名为 user
    product: productModule // 挂载商品模块，命名为 product
  }
})
```

## 3、Vuex的装载

- 首先导入module,并创建一个Vuex实例store

``` vue
import { createStore } from 'vuex'

// 导入模块（若有）
import user from './modules/user'
import product from './modules/product'

const store = createStore({
  modules: { user, product }
})

export default store
```

- 将Vuex挂载在App上

``` vue
import { createApp } from 'vue'
import App from './App.vue'
import store from './store'

const app = createApp(App)
createApp(App).use(store).mount('#app')
```

现在Vuex就已经成功挂载上去了，现在还有一个更加先进的状态管理器叫做Pinia，可以自行了解

 

# 六、路由

现在我们已经能够实现一个页面了，不过我们不能只做“单页面战士”,而页面的切换则需要通过路由实现

## 1、安装Router

首先要安装Vue3对应的Router

``` npm install vue-router@4
npm install vue-router
```

在src路径下创建router文件夹，将index.js放在其下，这里就是你的全局路由

## 2、路由配置

这里通过几个例子来展示一下路由的配置

### 路径配置

通过这种方式可以通过修改路径来访问相应的组件

``` vue
import Vue from 'vue'
import Router from 'vue-router'
import Home from '../components/Home.vue'
import About from '../components/About.vue'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',//这是根路径，或者说初始路径？
      name: 'Home',
      component: Home
    },
    {
      path: '/about',
      name: 'About',
      component: About
    }
  ]
})
```

### 点击跳转

下方给出了示例如何实现点击跳转,有两种：一种是通过router-link，另外一种是通过push方法

``` vue
<template>
  <!-- 基本用法：跳转到指定路径 -->
  <router-link to="/home">首页</router-link>

  <!-- 跳转到命名路由（需在路由配置中定义 name） -->
  <router-link :to="{ name: 'About' }">关于我们</router-link>

  <!-- 带参数的跳转 -->
  <router-link :to="{ name: 'User', params: { id: 123 } }">
    用户 123 的详情
  </router-link>

  <!-- 带查询参数（?page=1） -->
  <router-link :to="{ path: '/list', query: { page: 1 } }">
    列表第 1 页
  </router-link>
</template>
```

```vue
<template>
  <!-- 绑定点击事件 -->
  <button @click="goToHome">跳转到首页</button>
  <button @click="goToUser(456)">跳转到用户 456</button>
</template>

<script setup>
import { useRouter } from 'vue-router'

// 获取路由实例
const router = useRouter()

// 跳转到首页
const goToHome = () => {
  router.push('/home')
}

// 带参数跳转
const goToUser = (userId) => {
  router.push({ 
    name: 'User', 
    params: { id: userId } 
  })
}
</script>
```

### 嵌套路由

嵌套路由意思是指在一个路由中嵌套了一个完整路由，嵌套路由的运用很广泛

首先来个最简单的嵌套路由的例子

```Vue
import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import User from '../views/User.vue'
import UserProfile from '../views/user/Profile.vue'
import UserOrders from '../views/user/Orders.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/user',  // 父路由路径
    name: 'User',
    component: User,
    // 子路由配置（嵌套路由的核心）
    children: [
      {
        // 子路由路径为空时，作为默认显示
        path: '',
        redirect: 'profile'  // 默认显示个人信息页面
      },
      {
        // 注意：子路由路径不要加 /，最终路径会自动拼接为 /user/profile
        path: 'profile',
        name: 'UserProfile',
        component: UserProfile
      },
      {
        path: 'orders',
        name: 'UserOrders',
        component: UserOrders
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
```

接下来我要说一个情况，当你的项目足够大的时候，你不能套个成百上千个吧

在你的项目中，肯定会存在一个根路由，但如果将所有路由都写在这个根路由下，就会显得十分冗杂。这个时候将每个组件的路由独立出来作为一个整体路由变量，然后将其放在根路由下，这样就成功将路由模块化设计

下面是一个简单的电商网页项目结构示例

```plaintext
src/
├── router/
│   ├── index.js          # 根路由配置（汇总所有模块）
│   ├── modules/          # 路由模块目录
│   │   ├── home.js       # 首页相关路由
│   │   ├── user.js       # 用户中心相关路由
│   │   ├── product.js    # 商品相关路由
│   │   └── order.js      # 订单相关路由
```

下面的示例的index.js的文件

```Vue
import { createRouter, createWebHistory } from 'vue-router'

// 导入各模块路由
import homeRoutes from './modules/home'
import userRoutes from './modules/user'
import productRoutes from './modules/product'
import orderRoutes from './modules/order'
import commonRoutes from './modules/common'

// 使用 ... 合并所有路由
const routes = [
  // 公共路由（登录、首页等）
  ...commonRoutes,
  
  // 业务模块路由
  ...homeRoutes,
  ...userRoutes,
  ...productRoutes,
  ...orderRoutes,
  
  // 404 兜底路由
  {
    path: '/:pathMatch(.*)*',
    redirect: '/404'
  }
]

// 创建路由实例
const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  // 切换路由时自动滚动到顶部
  scrollBehavior: () => ({ top: 0 })
})

export default router

```

当然还有一个更便捷的版本，不需要你手动去导入

```Vue
import { createRouter, createWebHistory } from 'vue-router'

// 自动导入 modules 目录下的所有路由模块
const modulesFiles = import.meta.glob('./modules/*.js', { eager: true })

// 收集所有路由
const routes = []
Object.values(modulesFiles).forEach((module) => {
  // 每个模块默认导出的是路由数组
  if (module.default && Array.isArray(module.default)) {
    routes.push(...module.default)
  }
})

// 创建路由实例
const router = createRouter({
  routes,
})

export default router
```

## 3、路由守卫

路由守卫是 Vue Router 提供的一种钩子函数机制，用于在路由跳转过程中进行拦截和控制，常用于常用来处理登录验证、权限检查、页面跳转限制、加载状态等场景。

就相当于一个门卫，在路由跳转前后进行状态检查或者执行其他状态

### 全局守卫

定义在路由实例上，对所有路由生效

```Vue
const router = createRouter({ ... })

// 1. 全局前置守卫（跳转前触发）
router.beforeEach((to, from, next) => {
  // to: 即将进入的目标路由
  // from: 当前要离开的路由
  // next: 必须调用的函数，决定是否继续跳转
  if (to.path === '/admin' && !isLogin()) {
    // 未登录访问管理员页面，强制跳转到登录页
    next('/login')
  } else {
    // 允许跳转
    next()
  }
})

// 3. 全局后置钩子（跳转后触发，无next函数）
router.afterEach((to, from) => {
  console.log("afterEach")
})
```

### 独享守卫

在路由配置中定义，只对当前路由生效

```vue
const routes = [
  {
    path: '/user',
    component: User,
    // 路由独享守卫（只针对当前路由的前置检查）
    beforeEnter: (to, from, next) => {
      // 例如：检查用户是否完成实名认证
      if (!isRealNameAuthenticated()) {
        next('/user/auth')
      } else {
        next()
      }
    }
  }
]
```

### 组件内守卫

在组件中定义，控制组件进入 / 离开的逻辑

``` vue
<script>
export default {
  // 1. 进入组件前触发
  beforeRouteEnter(to, from, next) {
    // 注意：此时组件实例还未创建（this 不可用）
    // 可以通过 next(vm => { ... }) 访问组件实例
    next(vm => {
      // 例如：获取组件数据
      vm.loadData()
    })
  },

  // 2. 路由参数变化时触发
  beforeRouteUpdate(to, from, next) {
    // 此时 this 可用
    this.userId = to.params.id
    this.loadData()
    next()
  },

  // 3. 离开组件时触发
  beforeRouteLeave(to, from, next) {
    // 例如：确认是否保存未提交的表单
    if (this.isFormDirty) {
      if (confirm('数据未保存，确定离开吗？')) {
        next()
      } else {
        next(false) // 取消跳转
      }
    } else {
      next()
    }
  }
}
</script>
```



# 七、结语

本篇博客只是个人学习Vue之后对基础知识的总结，不保证完整与完全正确，学习了这些你已经具备了编写Vue项目的能力，但是学无止境，Vue也是与时俱进的，许多前沿的技术需要我们在未来的日子里共同探索

自强不息。

