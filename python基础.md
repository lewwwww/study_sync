# **python基础**

###### 返回对象的内存地址

```
>>> class Student():
      def __init__(self,id,name):
        self.id = id
        self.name = name
          
>>> xiaoming = Student('001','xiaoming') 
>>> id(xiaoming)
2281930739080
```

###### 返回对象的哈希值

自定义的实例都可哈希，`list`, `dict`, `set`等可变对象都不可哈希(unhashable)：

```
>>> class Student():
      def __init__(self,id,name):
        self.id = id
        self.name = name
        
>>> xiaoming = Student('001','xiaoming')
>>> hash(xiaoming)
-9223371894234104688
```

###### 排序

```
>>> a = [1,4,2,3,1]
#降序
>>> sorted(a,reverse=True)
[4, 3, 2, 1, 1]
>>> a = [{'name':'xiaoming','age':18,'gender':'male'},
       {'name':'xiaohong','age':20,'gender':'female'}]
#按 age升序
>>> sorted(a,key=lambda x: x['age'],reverse=False)
[{'name': 'xiaoming', 'age': 18, 'gender': 'male'}, 
{'name': 'xiaohong', 'age': 20, 'gender': 'female'}]
```

###### print用法

```
>>> lst = [1,3,5]
# f 打印
>>> print(f'lst: {lst}')
lst: [1, 3, 5]
# format 打印
>>> print('lst:{}'.format(lst))
lst:[1, 3, 5]
```

###### 字符串格式化

```python
>>> print("i am {0},age {1}".format("tom",18))
i am tom,age 18
>>> print("{:.2f}".format(3.1415926)) # 保留小数点后两位
3.14
>>> print("{:+.2f}".format(-1)) # 带符号保留小数点后两位
-1.00
>>> print("{:.0f}".format(2.718)) # 不带小数位
3
>>> print("{:0>3d}".format(5)) # 整数补零，填充左边, 宽度为3
005
>>> print("{:,}".format(10241024)) # 以逗号分隔的数字格式
10,241,024
>>> print("{:.2%}".format(0.718)) # 百分比格式
71.80%
>>> print("{:.2e}".format(10241024)) # 指数记法
1.02e+07
```

###### 打开文件

```python
>>> import os
>>> os.chdir('D:/source/dataset')
>>> os.listdir()
['drinksbycountry.csv', 'IMDB-Movie-Data.csv', 'movietweetings', 
'titanic_eda_data.csv', 'titanic_train_data.csv']
>>> o = open('drinksbycountry.csv',mode='r',encoding='utf-8')
>>> o.read()
"country,beer_servings,spirit_servings,wine_servings,total_litres_of_pur
e_alcohol,continent\nAfghanistan,0,0,0,0.0,Asia\nAlbania,89,132,54,4.9,"
```

###### **with 读写文件**

```
>> import os
>>> os.chdir('D:/source/dataset')
>>> os.listdir()
['drinksbycountry.csv', 'IMDB-Movie-Data.csv', 'movietweetings', 'test.csv', 'titanic_eda_data.csv', 'titanic_train_data.csv', 'train.csv']
# 读文件
>>> with open('drinksbycountry.csv',mode='r',encoding='utf-8') as f:
      o = f.read()
      print(o)
      
# 写文件
>>> with open('new_file.txt',mode='w',encoding='utf-8') as f:
      w = f.write('I love python\n It\'s so simple')
      os.listdir()

 
['drinksbycountry.csv', 'IMDB-Movie-Data.csv', 'movietweetings', 'new_file.txt', 'test.csv', 'titanic_eda_data.csv', 'titanic_train_data.csv', 'train.csv']
>>> with open('new_file.txt',mode='r',encoding='utf-8') as f:
      o = f.read()
      print(o)
 
I love python
 It's so simple
```

######  **提取后缀名**

```
>>> import os
>>> os.path.splitext('D:/source/dataset/new_file.txt')
('D:/source/dataset/new_file', '.txt') #[1]：后缀名
```

###### **提取完整文件名**

```
>>> import os
>>> os.path.split('D:/source/dataset/new_file.txt')
('D:/source/dataset', 'new_file.txt')
```



###### 两种创建属性方法

```
>>> class C:
    def __init__(self):
      self._x = None
    def getx(self):
      return self._x
    def setx(self, value):
      self._x = value
    def delx(self):
      del self._x
    # 使用property类创建 property 属性
    x = property(getx, setx, delx, "I'm the 'x' property.")



使用C类
>>> C().x=1
>>> c=C()
# 属性x赋值
>>> c.x=1
# 拿值
>>> c.getx()
1
# 删除属性x
>>> c.delx()
# 再拿报错
>>> c.getx()
Traceback (most recent call last):
  File "<pyshell#118>", line 1, in <module>
    c.getx()
  File "<pyshell#112>", line 5, in getx
    return self._x
AttributeError: 'C' object has no attribute '_x'
# 再属性赋值
>>> c.x=1
>>> c.setx(1)
>>> c.getx()
1
```

使用`@property`装饰器，实现与上完全一样的效果：

```
class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
```

动态删除属性

```
>>> class Student():
      def __init__(self,id,name):
        self.id = id
        self.name = name

>>> xiaoming = Student('001','xiaoming')
>>> delattr(xiaoming,'id')
>>> hasattr(xiaoming,'id')
False
```

一键查看对象所有方法

```
>>> class Student():
      def __init__(self,id,name):
        self.id = id
        self.name = name

>>> xiaoming = Student('001','xiaoming')
>>> dir(xiaoming)
['__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'id', 'name']
```

枚举对象

```
>>> s = ["a","b","c"]
>>> for i,v in enumerate(s):
       print(i,v)
0 a
1 b
2 c
```

创建迭代器

```
>>> class TestIter():
 def __init__(self,lst):
  self.lst = lst
  
 # 重写可迭代协议__iter__
 def __iter__(self):
  print('__iter__ is called')
  return iter(self.lst)
  
  迭代 TestIter 类：
>>> t = TestIter()
>>> t = TestIter([1,3,5,7,9])
>>> for e in t:
 print(e)

 
__iter__ is called
1
3
5
7
9
```

创建range迭代器

1. range(stop)

2. range(start, stop[,step])

   ```
   >>> t = range(11)
   >>> t = range(0,11,2)
   >>> for e in t:
        print(e)
   
   0
   2
   4
   6
   8
   10
   ```

   打包

   ```
   >>> x = [3,2,1]
   >>> y = [4,5,6]
   >>> list(zip(y,x))
   [(4, 3), (5, 2), (6, 1)]
   >>> for i,j in zip(y,x):
    print(i,j)
   
   4 3
   5 2
   6 1
   ```

   过滤器

   ```
   >>> fil = filter(lambda x: x>10,[1,11,2,45,7,6,13])
   >>> for e in fil:
          print(e)
   
   11
   45
   13
   ```

   链式比较， Python会按照从左到右的顺序计算

   ```
   >>> i = 3
   >>> 1 < i < 3
   False
   >>> 1 < i <=3
   True
   ```

# python核心

###### 斐波那契数列前n项

```
>>> def fibonacci(n):
      a, b = 1, 1
      for _ in range(n):
        yield a
        a, b = b, a+b # 注意这种赋值

>>> for fib in fibonacci(10):
      print(fib)

 
1
1
2
3
5
8
13
21
34
55
```

###### **list 等分 n 组**

```
>>> from math import ceil
>>> def divide_iter(lst, n):
      if n <= 0:
        yield lst
        return
      i, div = 0, ceil(len(lst) / n)
      while i < n:
        yield lst[i * div: (i + 1) * div]
        i += 1

  
>>> for group in divide_iter([1,2,3,4,5],2):
      print(group)

 
[1, 2, 3]
[4, 5]
```

###### **装饰器**

```
from functools import wraps
import time
定义一个装饰器：print_info，装饰器函数入参要求为函数，返回值要求也为函数
def print_info(f):
    """
    @para: f, 入参函数名称
    """
    @wraps(f) # 确保函数f名称等属性不发生改变
    def info():
        print('正在调用函数名称为： %s ' % (f.__name__,))
        t1 = time.time()
        f()
        t2 = time.time()
        delta = (t2 - t1)
        print('%s 函数执行时长为：%f s' % (f.__name__,delta))

    return info
    
使用 print_info 装饰器，分别修饰 f1, f2 函数
@print_info
def f1():
    time.sleep(1.0)


@print_info
def f2():
    time.sleep(2.0)
    
使用装饰后的函数
f1()
f2()

# 输出信息如下：

# 正在调用函数名称为：f1
# f1 函数执行时长为：1.000000 s
# 正在调用函数名称为：f2
# f2 函数执行时长为：2.000000 s
```

迭代器案例

```
class YourRange():
    def __init__(self, start, end):
        self.value = start
        self.end = end

    # 成为迭代器类型的关键协议
    def __iter__(self):
        return self

    # 当前迭代器状态(位置)的下一个位置
    def __next__(self):
        if self.value >= self.end:
            raise StopIteration

        cur = self.value
        self.value += 1
        return cur
        
yr = YourRange(5, 12)
for e in yr:
    print(e)
```

#####  4 **种常见的绘图库绘制柱状图和折线图**

###### matplotlib

```
import matplotlib 
matplotlib.__version__  # '2.2.2'

import matplotlib.pyplot as plt 
plt.plot([0, 1, 2, 3, 4, 5],
        [1.5, 1, -1.3, 0.7, 0.8, 0.9]
        ,c='red')
plt.bar([0, 1, 2, 3, 4, 5],
        [2, 0.5, 0.7, -1.2, 0.3, 0.4]
        )
plt.show()
```

![image-20240710150534644](C:\Users\gjy\AppData\Roaming\Typora\typora-user-images\image-20240710150534644.png)

###### **seaborn**

```
import seaborn as sns 
sns.__version__ # '0.8.0'

sns.barplot([0, 1, 2, 3, 4, 5],
        [1.5, 1, -1.3, 0.7, 0.8, 0.9]
        )
sns.pointplot([0, 1, 2, 3, 4, 5],
        [2, 0.5, 0.7, -1.2, 0.3, 0.4]
        )
plt.show()
```

![image-20240710150607652](C:\Users\gjy\AppData\Roaming\Typora\typora-user-images\image-20240710150607652.png)

###### plotly 绘图

```
import plotly 
plotly.__version__ # '2.0.11'
import plotly.graph_objs as go
import plotly.offline as offline

pyplt = offline.plot
sca = go.Scatter(x=[0, 1, 2, 3, 4, 5],
             y=[1.5, 1, -1.3, 0.7, 0.8, 0.9]
            )
bar = go.Bar(x=[0, 1, 2, 3, 4, 5],
            y=[2, 0.5, 0.7, -1.2, 0.3, 0.4]
            )
fig = go.Figure(data = [sca,bar])
pyplt(fig)
```

![image-20240710150744618](C:\Users\gjy\AppData\Roaming\Typora\typora-user-images\image-20240710150744618.png)

###### pyecharts

```
import pyecharts
pyecharts.__version__ # '1.7.1'
bar = (
        Bar()
        .add_xaxis([0, 1, 2, 3, 4, 5])
        .add_yaxis('ybar',[1.5, 1, -1.3, 0.7, 0.8, 0.9])
    )
line = (Line()
        .add_xaxis([0, 1, 2, 3, 4, 5])
        .add_yaxis('yline',[2, 0.5, 0.7, -1.2, 0.3, 0.4])
        )
bar.overlap(line)
bar.render_notebook()
```

![image-20240710150924651](C:\Users\gjy\AppData\Roaming\Typora\typora-user-images\image-20240710150924651.png)

###### meshgrid 3D 曲面图，创建网格点

```
import numpy as np
import matplotlib.pyplot as plt

nx, ny = (5, 3)
x = np.linspace(0, 1, nx)
x
# 结果
# array([0.  , 0.25, 0.5 , 0.75, 1.  ])

y = np.linspace(0, 1, ny)
y 
# 结果
# array([0. , 0.5, 1. ])

xv, yv = np.meshgrid(x, y)
xv
xv 结果：
array([[0.  , 0.25, 0.5 , 0.75, 1.  ],
       [0.  , 0.25, 0.5 , 0.75, 1.  ],
       [0.  , 0.25, 0.5 , 0.75, 1.  ]])
yv 结果：       
array([[0. , 0. , 0. , 0. , 0. ],
       [0.5, 0.5, 0.5, 0.5, 0.5],
       [1. , 1. , 1. , 1. , 1. ]])
绘制网格点：       
plt.scatter(xv.flatten(),yv.flatten(),c='red')
plt.xticks(ticks=x)
plt.yticks(ticks=y)
```

###### 绘制曲面图

```
from mpl_toolkits.mplot3d import Axes3D
# X, Y 
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)    # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# Z
Z = np.sin(R)
fig = plt.figure()
ax = Axes3D(fig)
plt.xticks(ticks=np.arange(-5,6))
plt.yticks(ticks=np.arange(-5,6))
ax.plot_surface(X, Y, Z, cmap=plt.get_cmap('rainbow'))
plt.show()
```

![image-20240710170102183](C:\Users\gjy\AppData\Roaming\Typora\typora-user-images\image-20240710170102183.png)

###### 等高线图：

以上 3D 曲面图的在 xy平面、 xz平面、yz平面投影，即是等高线图。

```
fig = plt.figure()
ax = Axes3D(fig)
plt.xticks(ticks=np.arange(-5,6))
plt.yticks(ticks=np.arange(-5,6))
ax.contourf(X, Y, Z, zdir='z', offset=-1, cmap=plt.get_cmap('rainbow'))
plt.show()
```

![image-20240710170424798](C:\Users\gjy\AppData\Roaming\Typora\typora-user-images\image-20240710170424798.png)

# python习惯

###### **/ 返回浮点数**

```
In [1]: 8/5
Out[1]: 1.6
```

###### **// 得到整数部分**

```
In [2]: 8//5
Out[2]: 1

In [3]: a = 8//5
In [4]: type(a)
Out[4]: int
```

######  **% 得到余数**

###### ***\* 计算乘方**

###### **交互模式下的_**

```
In [8]: 2*3.02+1
Out[8]: 7.04

In [9]: 1+_
Out[9]: 8.04
```

###### **单引号和双引号微妙不同**

```
使用一对双引号时，打印下面串无需转义字符
In [10]: print("That isn't a horse")
That isn't a horse
使用单引号时，需要添加转义字符 \：
In [11]: print('That isn\'t a horse')
That isn't a horse
```

###### **跨行连续输入**

符串字面值可以跨行连续输入；一种方式是用一对三重引号：`"""` 或 `'''`

###### **数字和字符串**

```
In [13]: 3*'Py'
Out[13]: 'PyPyPy'
```

###### **连接字面值**

```
In [14]: 'Py''thon'
Out[14]: 'Python'
```

######  **if not x**

```
x = [1,3,5]

if x:
    print('x is not empty ')

if not x:
    print('x is empty')
```

###### **enumerate 枚举**

```
x = [1, 3, 5]

for i, e in enumerate(x, 10): # 枚举
    print(i, e)
```

###### **in**

```
x = 'zen_of_python'
if 'zen' in x:
    print('zen is in')
```

######  **zip 打包**

使用 zip 打包后结合 for 使用输出一对，更加符合习惯

```
keys = ['a', 'b', 'c']
values = [1, 3, 5]

for k, v in zip(keys, values):
    print(k, v)
```

###### **一对 '''**

```
print('''"Oh no!" He exclaimed.
"It's the blemange!"''')
```

###### **交换元素**

```
a, b = 1, 3
a, b = b, a  # 交换a,b
```

######  **join 串联**

```
chars = ['P', 'y', 't', 'h', 'o', 'n']
name = ''.join(chars)
print(name)
```

###### **列表生成式**

```
data = [1, 2, 3, 5, 8]
result = [i * 2 for i in data if i & 1] # 奇数则乘以2
print(result) # [2, 6, 10]
```

######  **字典生成式**

```
keys = ['a', 'b', 'c']
values = [1, 3, 5]

d = {k: v for k, v in zip(keys, values)}
print(d)
```

**`__name__ == '__main__'`有啥用**

导入包时有用

###### **字典默认值**

```
In[1]: d = {'a': 1, 'b': 3}

In[2]: d.get('b', [])  # 存在键 'b'
Out[2]: 3

In[3]: d.get('c', [])  # 不存在键 'c'，返回[]
Out[3]: []
```

######  **lambda 函数**

```
x = [1, 3, -5]
y = max(x, key=lambda x: abs(x))
print(y) # -5 
```

###### **map**

```
x = map(str, [1, 3, 5])
for e in x:
    print(e, type(e))# '1','3','5'
```

###### **reduce**

```
from functools import reduce
x = [1, 3, 5]
y = reduce(lambda p1, p2: p1*p2, x)
print(y) # 15
```

###### **filter**

```
x = [1, 2, 3, 5]
odd = filter(lambda e: e % 2, x)
for e in odd:  # 找到奇数
    print(e)
#另一种方法
odd = [e for e in x if e % 2]
print(odd) # [1,3,5]
```

