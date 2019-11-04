# 拼音输入法作业

## Blog

由于这里无法显示 LaTeX，为更好的阅读体验，请浏览 [siahuat0727 Blog](https://siahuat0727.github.io/2019/11/01/pinyin-input-method/)。

## 介绍

输入拼音，输出汉字。
```
src$ python main.py --load-model ../model/model.pkl 
qing hua da xue
清华大学
```
详见[这里](https://github.com/siahuat0727/PinYin/blob/master/拼音输入法编程作业.pdf)。

**提供**
1. `training_data/sina_new_gbk/`：训练语料文件夹，2016年**新浪新闻**部分内容。（约 4 亿字）
2. `training_data/拼音汉字表.txt` ：拼音汉字映射表。e.g. `a 啊 嗄 腌 吖 阿 锕`
3. `training_data/一二级汉字表.txt`：汉字表。e.g. `啊阿埃挨哎唉哀皑癌蔼矮艾碍爱...`

**测试**
1. `testing_data/news.txt`：测试文字（答案），[爬虫教学](https://blog.csdn.net/qq_33722172/article/details/82469050)爬近期新浪新闻+预处理。（约 8 万字）
2. `testing_data/input.txt`：测试拼音（题目），[汉字拼音在线转换](https://www.qqxiuzi.cn/zh/pinyin/)+预处理。
3. 准确率计算：逐字计算。


**代码功能**（均通过命令行参数）
1. 只要资源允许，可训练**任意** n 元字模型（n>=2）。
2. 可依据需求在准确率和速度上做出相应的折中（简化版DP，完整版DP，加速×简化版DP，加速×完整版DP）。

**成果**
1. `testing_data/predict.txt`：4 元加速x完整版 DP [预测结果](https://github.com/siahuat0727/PinYin/blob/master/testing_data/predict.txt) v.s. [正确答案](https://github.com/siahuat0727/PinYin/blob/master/testing_data/news.txt)（字准确率 94.82%）

## Usage

```shell
src$ python main.py --help

usage: main.py [-h] [--verbose] [--input-file INPUT_FILE]
               [--output-file OUTPUT_FILE] [--load-model LOAD_MODEL]
               [--save-model SAVE_MODEL] [--words WORDS]
               [--pinyin-table PINYIN_TABLE] [--init-words]
               [--init-pinyin-table] [--train] [--analysis]
               [--encoding ENCODING] [--file FILE] [--dir DIR] [--match MATCH]
               [--alpha ALPHA] [--n-gram N_GRAM] [--no-clip] [--brute-force]
               [--fast] [--slim] [--threshold THRESHOLD]

PinYin Input Method

optional arguments:
  -h, --help            show this help message and exit
  --verbose             whether to print more information (default: False)
  --input-file INPUT_FILE
                        input file (if any) (default: None)
  --output-file OUTPUT_FILE
                        output file (if any) (default: None)
  --load-model LOAD_MODEL
                        path to load model (default: None)
  --save-model SAVE_MODEL
                        path to save model (default: ../model/model.pkl)
  --words WORDS         path to save words (default: ../model/words.pkl)
  --pinyin-table PINYIN_TABLE
                        path to save pinyin-table (default:
                        ../model/pinyin_table.pkl)
  --init-words          task: init words (default: False)
  --init-pinyin-table   task: init pinyin-table (default: False)
  --train               task: train model (default: False)
  --analysis            task: analysis model (default: False)
  --encoding ENCODING   input file coding method (default: utf8)
  --file FILE           path to file (default: None)
  --dir DIR             path to dir (default: None)
  --match MATCH         regex to match training files when given directory
                        (default: .*)
  --alpha ALPHA         smoothing factor (default: 0.9)
  --n-gram N_GRAM       using n-gram model (default: 2)
  --no-clip             disable clip for words (check README.md for detail)
                        (default: False)
  --brute-force         use brute force (instead of dynamic programming)
                        (default: False)
  --fast                only find approx. answer when using dynamic
                        programming (default: False)
  --slim                make pinyin-table slimmer (default: False)
  --threshold THRESHOLD
                        del a word from pinyin-table if # of the word is less
                        than threshold (default: 100)
```

## 复现步骤+使用方法

#### 0. 下载语料库（2016年部分新浪新闻）
```
training_data$ curl -fLo data.zip https://cloud.tsinghua.edu.cn/f/311cdd5c41404b9a940a/?dl=1
...

training_data$ unzip data.zip
Archive:  data.zip
   creating: sina_news_gbk/
  inflating: sina_news_gbk/2016-09.txt
  ...
  
training_data$ rm data.zip
```

*注：里面的文件都是 gbk 编码*

> 就不吐槽里面的 README.txt 用 gbk 编码说大家是 utf-8 了（（（

#### 1. 建立中文汉字库
`python main.py --init-words --file FILE [--encoding ENCODING] [--words WORDS] [--verbose]`

```
src$ python main.py --init-words --file ../training_data/一二级汉字表.txt --encoding gbk --verbose

Loaded ../training_data/一二级汉字表.txt 
Saved ../model/words.pkl
```

#### 2. 建立拼音汉字映射表
`python main.py --init-pinyin-table --file FILE [--encoding ENCODING] [--pinyin-table PINYIN_TABLE] [--verbose]`
```
src$ python main.py --init-pinyin-table --file ../training_data/拼音汉字表.txt --encoding gbk --verbose

Loaded ../training_data/拼音汉字表.txt 
Saved ../model/pinyin_table.pkl
```

#### 3. 训练模型 
`python main.py --train [--file FILE] [--dir DIR] [--match MATCH] [--encoding gbk] [--load-model LOAD_MODEL] [--save-model SAVE_MODEL] [--words WORDS] [--pinyin-table PINYIN_TABLE] [--n-gram N_GRAM] [--verbose]`

`--file FILE`：单档训练。

`--dir DIR`：多档训练。可用 `--match <regular expression>` 来对应该目录底下符合规则的文件。

`--n-gram N_GRAM`：支援训练 n 字元模型，n >= 2。模型向下兼容，如 3 元模型可当 2 元模型用。

`--load-model LOAD_MODEL`：可基于已训练的模型继续训练。

```
src$ python main.py --train --dir ../training_data/sina_news_gbk/ --encoding gbk --match "^2016"  --verbose

Loaded ../model/words.pkl
Training files are:
	../training_data/sina_news_gbk/2016-04.txt
	../training_data/sina_news_gbk/2016-10.txt
	../training_data/sina_news_gbk/2016-06.txt
	../training_data/sina_news_gbk/2016-05.txt
	../training_data/sina_news_gbk/2016-11.txt
	../training_data/sina_news_gbk/2016-09.txt
	../training_data/sina_news_gbk/2016-08.txt
	../training_data/sina_news_gbk/2016-07.txt
	../training_data/sina_news_gbk/2016-02.txt
Start training for ../training_data/sina_news_gbk/2016-04.txt
78961 news trained

...

Start training for ../training_data/sina_news_gbk/2016-02.txt
67919 news trained
Saved ../model/model.pkl
```

#### 4. 分析模型

`python main.py --analysis --load-model LOAD_MODEL [--verbose]`

下文有例子。

#### 5. 执行任务

`python main.py --load-model LOAD_MODEL [--n-gram N_GRAM] [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--words WORDS] [--pinyin-table PINYIN_TABLE] [--encoding ENCODING] [--alpha ALPHA] [--brute-force] [--fast] [--slim] [--threshold] [--verbose]`

`--n-gram N_GRAM`：选择用 n 元模型来计算。`n >= 2`

`--input-file INPUT_FILE`：可给输入文件（预设stdin），并给相应的`--encoding ENCODING`

`--output-file OUTPUT_FILE`：可给输出文件（预设stdout）

`--alpha ALPHA`：即平滑系数 $\lambda$，详见本文。`0 <= alpha <= 1`

`--brute-force`：用暴力法来搜索。

`--fast`：用简化版动态规划。

`--slim`：启动删除部分汉字的功能，用于加速（详见`--threshold`）

`--threshold THRESHOLD`：删除在训练语料中出现次数少于 `threshold` 的汉字

[介绍](#介绍) 有例子。

## 算法

### 概率计算

利用条件概率的公式，句子$w_1w_2...w_m$ 出现的概率可展开为，
$P(w_1, w_-2, ..., w_m) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdot \dots \cdot  P(w_n|w_1, w_2,...,w_{n-1})$

但条件概率 $P(w_n|w_1, w_2,...,w_{n-1})$ 的可能性太多，无法估算，于是数学家马尔科夫提出一个简单但很有效的假设，即任意一个字 $w_i$ 的出现只与它前面的字 $w_{i-1}$ 有关，于是模型变得非常简单。

#### 2 元字模型

假设一个字 $w_i$ 的出现只与它前面的字 $w_{i-1}$ 有关。

句子 $w_1w_2...w_m$ 出现的概率
$$P(w_1,w_2,...,w_m) = \prod_{i=2}^m P(w_i|w_{i-1})$$

> 公式参考自作业 ppt

而$$P(w_i|w_{i-1}) = \frac{P(w_{i-1},w_i)}{P(w_{i-1})} \approx  \frac{\#(w_{i-1},w_i)}{\#w_{i-1}}$$

即 $(w_{i-1}, w_i)$ 同时在语料中出现的次数 #$(w_{i-1}, w_i)$ 与 $w_{i-1}$ 单独在语料中出现的次数 #$w_{i-1}$ 的比值。

这里只写了**非常简化**的计算公式，实际细节见[分析与优化](#分析与优化)。



#### n 元字模型

更普遍一点会假设每个字 $w_i$ 的出现和前面 n-1 个字有关，这种假设被称为 n-1 阶马尔科夫假设，对应的模型称为 n 元模型。


> **概率计算** 这一段参考自《数学之美》第 3 章

### 搜索方法

$m$：句子长度
$n$：$n$ 元字模型
$c$：每个拼音对应 $c$ 个汉字（简单假设）

$w_{i, j}$： 句子中第 $i^{th}$ 字的第 $j^{th}$ 候选字
$W_i$：句子中第 $i^{th}$ 字的候选字集合
$T_{i, j}$： 长度为 $i$，结尾为 $w_{i, j}$ 的最佳前缀
$T_{i, *}$： 长度为 $i$ 的最佳前缀

#### 暴力法

对所有可能组成的句子都搜索一遍。

时间复杂度：$O(c^m)$

#### 简化版动态规划

每个字 $w_i$ 只与前一个字有关，因此从不同 $w_{i-1}$ 结尾的最佳前缀来搜索即可。
即 $w_{i-2}$ 及之前的字不会影响到当前选择，因此无需考虑以 $w_{i-1}$ 结尾的非最佳前缀。

> 之所以是简化版，是因为在 n>2 的 n 元模型中这个方法找到的就不一定是 “最好” 的句子了，详见[完整版动态规划](#解决方案完整版动态规划)。


$$P(T_{i, j})  =
  \begin{cases}
\max_{k}(P(T_{i-1, k})\cdot P(w_{i,j}|T_{i-1, k})) & \quad  \text{if } i \geq 2\\
P(w_{i,j}) & \quad  \text{if } i = 1 
  \end{cases}$$
$$T_{i, *} = \mathop{\arg\max}_{T_i \in \{T_{i, 1}, T_{i, 2}, ..., T_{i, |W_i|}\}}P(T_i)$$

> 第一次自己写这类公式，不知道写得对不对清不清楚，有任何错误或建议欢迎指点啦！[GitHub Issue](https://github.com/siahuat0727/PinYin/issues)

目标：找到 $T_{m,*}$ 作为输出。

时间复杂度：$O(c \cdot m)$

## 分析与优化

### 2 元字模型

#### 1. 关于第一个字出现的概率

##### 现象

$$P(w_1,w_2,...,w_m) = \prod_{i=2}^m P(w_i|w_{i-1})$$

以上公式仅假设了一个字出现的概率与前一个字有关，对于第一个字不做任何限制。

这样显然是不行的，看以下例子：

```
src$ python main.py --load-model ../model/model-2-gram.pkl --analysis

Input 2 keys, output the number of times key1 followed by key2.
For example,
Input: 你 好
Output: 1234 (the number of times 你 followed by 好)
--------------------------------------------------
溘 埃
2
溘 total
56
可 爱
4965
可 total
817610
```
*注：查询 `some_word total` 将给出 `some_word` 出现的总次数，详见[关于 `total` 魔法](#2-关于-total-魔法)。*



于是 `ke ai` -> `溘埃`

> 溘埃： 2/56 = 0.0357  
> 可爱： 4965/817610 = 0.0061  
> 溘埃 > 可爱

##### 分析

主要是`溘`紧挨着的字选择不多，所以对于每个选择，出现的概率反而更高。
而`可爱`虽然很常见，但`可`后面可以接的字实在太丰富了，`爱`对`可`来说紧挨着的概率非常低。

> 驷玉虬以乘鹥兮，**溘埃**风余上征。  
> 出处：屈原《离骚》

##### 解决方案

对于第一个字，我们可以借用单字本身
