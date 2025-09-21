# MobileNetV1 详解

这几天学了一下轻量化网络，我们先从MobileNetV1讲起吧~

MobileNetV1网络是谷歌团队在2017年提出的，专注于移动端和嵌入设备的轻量级CNN网络，相比于传统的神经网络，在准确率小幅度降低的前提下大大减少模型的参数与运算量。相比于VGG16准确率减少0.9%，但模型的参数只有VGG1/32。

其实简单来说，就是把VGG中的标准卷积层换成深度可分离卷积。不过这个深度可分离卷积刚开始接触比较抽象难理解，建议大家看看文末链接里大佬的讲解视频噢！也欢迎大家在评论区讨论 (ฅ´ω`ฅ)~

## 学习资料

- **论文题目**：《MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications》（《MobileNets:用于移动视觉应用的高效卷积神经网络》）
- **原文地址**：https://arxiv.org/abs/1704.04861
- **项目地址**：https://github.com/Zehaos/MobileNet

## 目录

- [前言](#前言)
- [Abstract—摘要](#abstract摘要)
- [一、Introduction—简介](#一introduction简介)
- [二、Prior Work—先前的工作](#二prior-work先前的工作)
- [三、MobileNet Architecture— MobileNet结构](#三mobilenet-architecture--mobilenet结构)
  - [3.1 Depthwise Separable Convolution—深度可分离卷积](#31-depthwise-separable-convolution深度可分离卷积)
    - [(1)标准卷积](#1标准卷积)
    - [(2) depthwise convolution：深度卷积](#2--depthwise-convolution深度卷积)
    - [(3) pointwise convolution：逐点卷积](#3--pointwise-convolution逐点卷积)
  - [3.2 Network Structure and Training— 网络结构和训练](#32-network-structure-and-training--网络结构和训练)
  - [3.3 Width Multiplier: Thinner Models—宽度超参数α](#33-width-multiplier-thinner-models宽度超参数α)
  - [3.4 Resolution Multiplier: Reduced Representation—分辨率超参数ρ](#34-resolution-multiplier-reduced-representation分辨率超参数ρ)
- [四、Experiments—实验](#四experiments实验)
  - [4.1 Model Choices—模型的选择](#41-model-choices模型的选择)
  - [4.2 Model Shrinking Hyperparameters—模型收缩参数](#42-model-shrinking-hyperparameters模型收缩参数)
  - [4.3 Fine Grained Recognition—细粒度图像分类](#43-fine-grained-recognition细粒度图像分类)
  - [4.4 Large Scale Geolocalizaton—以图搜地](#44-large-scale-geolocalizaton以图搜地)
  - [4.5 Face Attributes—人脸属性](#45-face-attributes人脸属性)
  - [4.6 Object Detection— 目标检测](#46-object-detection--目标检测)
  - [4.7 Face Embeddings—人脸识别](#47-face-embeddings人脸识别)
- [五、Conclusion—结论](#五conclusion结论)
- [🌟代码实现](#代码实现)



## Abstract—摘要

### 翻译

我们提出了一类有效的模型称为移动和嵌入式视觉应用的移动网络。MobileNets是基于流线型架构，使用深度可分卷积来建立轻量级深度神经网络。我们介绍了两个简单的全局超参数，它们可以有效地在延迟和准确性之间进行权衡。这些超参数允许模型构建者根据问题的约束为其应用程序选择适当大小的模型。我们在资源和精度权衡方面进行了大量的实验，并与其他流行的ImageNet分类模型相比，显示了较强的性能。然后，我们演示了MobileNets在广泛的应用和用例中的有效性，包括目标检测、细粒度分类、人脸属性和大规模地理定位。

### 精读

#### 本文主要工作

1. 提出了一个可在移动端应用的高效网络MobileNets，其使用深度可分类卷积使网络轻量化同时保证精度。

2. 设计了两个控制网络大小全局超参数，通过这两个超参数来进行速度（时间延迟）和准确率的权衡，使用者可以根据设备的限制调整网络。

3. 在ImageNet分类集上与其他模型进行了广泛对比，验证了MobileNets的有效性。

## 一、Introduction—简介

### 翻译

自从AlexNet通过赢得ImageNet挑战赛:ILSVRC 2012普及了深度卷积神经网络以来，卷积神经网络在计算机视觉中变得无处不在。为了获得更高的精度，一般的趋势是构建更深更复杂的网络。然而，这些提高准确性的进步并不一定使网络在规模和速度方面更有效率。在机器人、自动驾驶汽车和增强现实等许多现实应用中，识别任务需要在一个计算能力有限的平台上及时执行。

本文描述了一种高效的网络架构和两个超参数，以便构建非常小的、低延迟的模型，可以很容易地满足移动和嵌入式视觉应用程序的设计要求。第2节回顾了以前在构建小模型方面的工作。第3节描述了MobileNet架构和两个超参数宽度乘法器和分辨率乘法器，以定义更小和更有效的MobileNet。第4节描述了在ImageNet上的实验以及各种不同的应用程序和用例。第5节以总结和结论结束。

### 精读

文章这一部分主要是简要介绍了一下卷积神经网络的发展和应用，然后引出自己提出的MobileNets网络，指出其在移动端应用方面的有效性，最后介绍本篇论文的写作框架。



## 二、Prior Work—先前的工作

### 翻译

在最近的文献中，人们对构建小型而高效的神经网络的兴趣越来越大。许多不同的方法可以大致分为压缩预训练网络和直接训练小型网络。本文提出了一类网络体系结构，它允许模型开发人员为其应用程序选择与资源限制(延迟、大小)匹配的小型网络。MobileNets主要关注于优化延迟，但也产生小的网络。许多关于小型网络的论文只关注规模而不考虑速度。mobilenet最初是由[26]中引入的深度可分卷积构建的，随后在Inception模型[13]中使用，以减少前几层的计算。扁平网络[16]构建了一个完全因数分解卷积的网络，并展示了高度因数分解网络的潜力。与本文无关，分解网络[34]引入了一个类似的分解卷积以及拓扑连接的使用。随后，Xception网络[3]演示了如何向上扩展深度可分离的过滤器，以执行Inception V3网络。另一个小型网络是Squeezenet[12]，它使用瓶颈方法来设计一个非常小的网络。其他简化计算网络包括结构变换网络[28]和油炸卷积网络[37]。

另一种获得小型网络的方法是缩小、分解或压缩预先训练好的网络。文献中提出了基于产品量化[36]、哈希[2]、剪枝、矢量量化和霍夫曼编码[5]的压缩方法。此外，还提出了各种因数分解方法来加速预先训练好的网络[14,20]。另一种训练小网络的方法是蒸馏[9]，它使用一个更大的网络来教一个更小的网络。它是对我们的方法的补充，并在第4节中介绍了我们的一些用例。另一种新兴的方法是低比特网络[4,22,11]。

### 精读

#### 以前的方法

- **压缩预训练网络**
- **直接训练小网络**

#### 相关轻量化网络

- **Inception网络**：使用深度可分离卷积减少前几层的计算量。
- **Flattened网络**：利用完全的因式分解的卷积网络构建模型，显示出完全分解网络的潜力。
- **Factorized Networks**：引入了类似的分解卷积以及拓扑连接的使用。
- **Xception网络**：演示了如何按比例扩展深度可分离卷积核。
- **Squeezenet网络**：使用一个bottleneck用于构建小型网络。

#### 建立小型高效的神经网络两种方法

1. **压缩预训练模型**：
   - 减小、分解或压缩预训练网络，例如量化压缩(product quantization)、哈希(hashing )、剪枝(pruning)、矢量编码( vector quantization)和霍夫曼编码(Huffman coding)等
   - 各种分解因子，用来加速预训练网络

2. **蒸馏**：
   - 使用大型网络指导小型网络

## 三、MobileNet Architecture— MobileNet结构

### 3.1 Depthwise Separable Convolution—深度可分离卷积

#### 翻译

标准的卷积层将作为输入DF×DF×M功能映射F和产生一个DF×DF×N特性图G, DF是输入特征图的高度和宽度,M是输入通道的数量(输入深度),DG是一个正方形的空间宽度和高度输出特性图和N是输出通道输出(深度)的数量。标准卷积层由大小为DK×DK×M×N的卷积核K参数化，其中DK为假设为平方的核的空间维数，M为输入通道数，N为前面定义的输出通道数。假设stride为1,padding为1，计算标准卷积的输出特征图为:

标准卷积的计算成本为:

其中，计算代价乘上输入通道M的数量，输出通道N的数量，核大小Dk×Dk, feature map大小DF×DF。MobileNet模型解决了这些术语及其相互作用。首先，它使用深度可分卷积来打破输出通道数量和内核大小之间的交互。标准卷积运算的作用是基于卷积核对特征进行滤波，并结合特征得到新的表示。滤波和组合步骤可分为两个步骤，通过使用分解卷积称为深度可分卷积，以大幅降低计算成本。深度可分卷积由两层构成:深度卷积和点卷积。我们使用深度卷积为每个输入通道应用一个过滤器(输入深度)。点态卷积，一个简单的1×1卷积，然后用来创建一个线性组合的输出的深度层。MobileNets对两个层都使用batchnorm和ReLU非线性。每个输入通道一个滤波器的深度卷积(输入深度)可表示为:

其中K^为大小为DK×DK×M的深度卷积核，其中K^中的第M个滤波器应用于F中的第M个信道，得到滤波后的输出特征映射G^的第M个信道。深度卷积的计算代价为:

深度卷积相对于标准卷积是非常有效的。但是，它只过滤输入通道，而不将它们组合起来创建新特性。因此，需要通过1×1的卷积来计算深度卷积输出的线性组合，以生成这些新特性。深度卷积和1×1(点态)卷积的组合称为深度可分离卷积，最早是在[26]中引入的。深度可分卷积的代价:

它是深度卷积和1×1点卷积的和。将卷积表示为滤波和组合的两步过程，计算量减少为:

MobileNet使用3×3深度可分卷积，它使用的计算量是标准卷积的8到9倍，仅在精度上略有降低，如第4节所示。空间维度上的额外因式分解，并没有节省很多额外的计算，因为深度卷积的计算量很少。

#### 精读

##### (1)标准卷积

标准卷积我们已经很熟悉了，直接看下图吧：

**方法**

标准卷积分成两步：

1. 使用卷积核对图中的特征进行提取。
2. 对提取的特征进行融合。

在标准卷积中这两步一般是同时进行的。

**特点**

- 卷积核channel=输入特征矩阵channel
- 输出特征矩阵channel=卷积核个数

**参数量**

Dk ×Dk ×M×N

**计算量**

Df×Df×Dk×Dk×M×N

其中Df为特征图尺寸，Dk为卷积核尺寸，M为输入通道数，N为输出通道数。

##### (2) depthwise convolution：深度卷积

深度卷积负责卷积作用，每个通道对应一个卷积核。

举个栗子：一个大小为64×64像素、3通道彩色图片首先经过第一次卷积运算，不同之处在于此次的卷积完全是在二维平面内进行的，且卷积核的数量与上一层的深度相同。所以一个3通道的图像经过运算后生成了3个Feature map，如下图所示

**方法**

深度分离卷积把输入特征图的所有通道进行分离，每个通道对应的一个卷积核对该通道的特征图进行单独的卷积操作(也就是说，第m个深度卷积核作用在输入的第m个通道上，得到输出结果的第m个通道)。在深度分离卷积中，每个卷积核的深度固定为1。

**特点**

- 卷积核channel=1
- 输入特征矩阵channel=卷积核个数=输出特征矩阵channel

**参数量**

Dk ×Dk ×M

**计算量**

Dk×Dk×M×Df×Df

其中Dk为卷积核尺寸，Df为特征图尺寸，M为输入通道数，输出通道数为1。

##### (3) pointwise convolution：逐点卷积

逐点卷积负责转换通道，是1x1卷积进行跨通道信息融合，可以大大减少参数量。

**方法**

Pointwise Convolution的运算与标准卷积运算非常相似，它的卷积核的尺寸为1×1×M，( M为上一层的通道数)。所以这里的卷积运算会将上一步的map在深度方向上进行加权组合，生成新的Feature map。有几个卷积核就有几个输出Feature map。

**参数量**

1 x 1 x M x N

**计算量**

1x 1 x M x N x Df x Df

其中卷积核尺寸是1×1，Df为特征图尺寸，M为输入通道数，N为输出通道数。

因此计算一次深度可分离卷积的总体计算量为:

它们减少计算量的比例(参数量比例同计算量)为:

深度卷积其实就是g=M=N的分组卷积，但没有将g组concate起来，所以参数量为标准卷积的1/N
逐点卷积其实就是g组卷积用con1×1拼接起来，所以参数量是标准卷积的1/Dk^2
3.2 Network Structure and Training— 网络结构和训练
翻译
MobileNet结构是建立在前一节提到的深度可分卷积，除了第一层是一个完整的卷积。通过用如此简单的术语定义网络，我们可以很容易地探索网络拓扑以找到一个好的网络。MobileNet体系结构定义在表1中。除了最后的全连接层不存在非线性，并输入softmax层进行分类之外，所有层都遵循batchnorm和ReLU非线性。图3对比了一层与正则卷积、batchnorm和ReLU的卷积后的深度卷积、1×1点卷积的分解层和每一层卷积后的batchnorm和ReLU的非线性关系。下行采样在深度卷积和第一层卷积中都使用了strided convolution。最终的平均池将空间分辨率降低到全连接层之前的1。将深度卷积和点卷积计算为独立的层，MobileNet有28层。

仅仅用少量的多添加来定义网络是不够的。确保这些操作能够有效地实现也很重要。例如，非结构化的稀疏矩阵操作通常不会比密集矩阵操作快，直到非常高的稀疏程度。我们的模型结构将几乎所有的计算都放在密集的1×1个卷积中。这可以通过高度优化的通用矩阵乘法(GEMM)函数来实现。通常，卷积是由GEMM实现的，但是需要在内存中进行名为im2col的初始重新排序才能将其映射到GEMM。例如，在流行的Caffe包[15]中使用了这种方法。1×1卷积不需要在内存中重新排序，可以直接使用GEMM实现，GEMM是最优化的数值线性代数算法之一。MobileNet将95%的计算时间花费在1×1个卷积上，其中75%的参数如表2所示。几乎所有的附加参数都在全连接层中。

MobileNet模型在TensorFlow[1]中使用RMSprop[33]进行训练，其异步梯度下降类似于Inception V3[31]。然而，与训练大型模型相反，我们使用较少的正则化和数据扩充技术，因为小型模型的过拟合问题较少。当训练MobileNets时，我们不使用侧头或标签平滑，并通过限制在大型初始训练[31]中使用的小型作物的大小来减少图像失真的数量。此外，我们发现在深度方向的滤波器上放很少或没有权值衰减(l2正则化)是很重要的，因为它们的参数很少。对于下一节中的ImageNet基准测试，所有模型都使用相同的训练参数进行训练，而不考虑模型的大小。

精读
(1)核心层(深度可分离卷积层)  

深度可分离卷积结合BN和ReLU，与标准卷积模型的对比如下



(2)整体网络结构

 

整个MobileNetV1网络除了平均池化层和softmax输出层外，共28层。

第1层为标准卷积，接下来26层为核心层结构(深度可分离卷积层)，最后是平均池化层，全连接层加softmax层输出。

除全连接层不使用激活函数，而使用softmax进行分类之外，其他所有层都使用BN和ReLU。

(3)训练设置

计算复杂度设计

深度可分离卷积结构几乎把全部的计算复杂度放到了1×1卷积中，在MobileNet中有95%的时间花费在1×1卷积上，这部分也占了75%的参数。



原因：

①1x1卷积核是一个比较密集的矩阵，所以在卷积计算时速度较快
②1x1卷积可以直接通过GEMM  (general matrix multiply)进行加速，其他类型的卷积在使用GEMM之前需要先经过Im2co
训练配置参数

使用Tensorflow框架训练，使用RMSprop优化器，异步梯度下降(并行数据)，需要“钞能力”才可以搞定整个结构。

注意事项： 由于小网络不易过拟合，MobileNet较少使用正则化和数据增强技术(如没有使用side heads或者label smoothing)。

3.3 Width Multiplier: Thinner Models—宽度超参数α
翻译
尽管基本的MobileNet体系结构已经很小而且延迟很低，但是很多时候一个特定的用例或应用程序可能需要模型更小和更快。为了构建这些较小和较昂贵的模型计算我们介绍一个非常简单的参数α称为宽度乘数。宽度乘数α的作用是在每一层薄网络统一。对于一个给定的乘数α层和宽度的输入通道数M变成αM和输出通道的数目N变成αN。切除分离卷积的计算成本与宽度乘数α是: 



α2 (0;1]典型设置为1、0.75、0.5、0.25。α= 1基线MobileNet和α< 1 MobileNets减少。宽度乘数降低计算成本的影响,参数的数量大约α2平方。宽度乘法器可以应用于任何模型结构，以定义一个新的更小的模型，具有合理的准确性、延迟和大小权衡。它用于定义一个新的简化结构，需要从零开始训练。

精读
目的

   为了构造这些更小、计算成本更低的模型，引入了一个非常简单的参数α，称为宽度乘数。

作用

   统一让网络中的每一层都更瘦

方法

   比如针对某一层网络和α ，输入通道从M变成αM ，输出通道从N变成αN，该层计算量变成了：



举个栗子：如果把α设置成0.8，那么网络结构中的每一个卷积层中的卷积核个数都变为0.8倍。

α=1时是基准的MobileNet，而α < 1时是减小的MobileNet。

计算量减少了



3.4 Resolution Multiplier: Reduced Representation—分辨率超参数ρ
翻译
第二个hyper-parameter减少神经网络的计算成本是一项决议乘数ρ。我们将其应用于输入图像，每一层的内部表示随后被相同的乘法器缩减。在实践中我们隐式地设置ρ通过设置输入分辨率。我们现在可以表达我们的网络的核心层次的计算成本与宽度切除可分离旋转乘数乘数ρα和解决:       





通常隐式设置，因此网络的输入分辨率为224、192、160或128。ρ= 1基线MobileNet和ρ< 1计算MobileNets减少。决议乘数的影响由ρ2降低计算成本。作为一个例子，我们可以看看一个典型的层在MobileNet和看到如何深度可分卷积，宽度乘子和分辨率乘子降低成本和参数。表3显示了一个层的计算和参数的数量，因为架构收缩方法依次应用于该层。第一行显示全卷积层的多加和参数，输入feature map大小为14×14×512，内核K大小为3×3×512×512。我们将在下一节详细讨论资源和准确性之间的权衡。

精读
作用

   分辨率超参数ρ用来减少每一层输出的特征图大小的，通过减小特征图的分辨率来降低模型所需要的计算量

方法

   ρ负责控制输入图像的尺寸，间接控制中间层feature map的大小。输入的尺寸大，中间层的feature map就大，feature map大卷积的次数就会变多，次数变多运算量变大。



计算量减少了



文中给出了标准卷积、深度可分离卷积以及添加了两种超参数深度可分离卷积的参数量对比例子，输入的feature map 为14x14x512，卷积核大小为3x3x512x512，乘法和加法的计算量如下图：



结论

通过对比看可以看到添加了两张超参数的网络计算量和权重参数量都相应减少
添加分辨率超参数ρ不影响权重的参数量(两个都是0.15)
四、Experiments—实验
4.1 Model Choices—模型的选择
翻译
首先，我们展示了深度可分卷积的MobileNet与全卷积模型的比较结果。在表4中我们可以看到，使用深度可分卷积与全卷积相比，在ImageNet上只降低了1%的精度，这在多加和参数上节省了很多。接下来，我们将展示使用宽度倍增器的较薄模型与使用较少层的较浅模型的比较结果。为了使MobileNet更浅，删除了表1中的5层可分离滤波器，其特征尺寸为14×14×512。表5显示，在相似的计算和参数数量下，使MobileNets变薄比变浅好3%。

精读
实验1：对比标准卷积和深度可分离卷积的效果

表4展示使用标准卷积和深度可分离卷积网络的效果对比



结论：虽然精读略有下降，但是使用深度可分离卷积的MobileNets计算量和参数小了很多。

实验2：在计算量恒定情况下，MobileNet是选择更瘦还是更浅的网络

表5展示一个浅层MobileNet  (删除了5层卷积层)和一个使用α等于0.75的MobileNet网络结构的对比



结论：可以看到二者的参数量和计算量差不多，但是深度减少之后的浅网络精度相对降低，所以超参数比减少网络层数有用。

4.2 Model Shrinking Hyperparameters—模型收缩参数
翻译
表6显示了精度,计算和尺寸缩小MobileNet架构的权衡宽度乘数α。精度下降平稳,直到架构是由25α=太小。

表7显示了通过训练具有降低输入分辨率的MobileNets，不同分辨率乘子的准确性、计算和大小权衡。精度随着分辨率的提高而平稳下降。

图4显示了之间的权衡ImageNet 16模型的准确性和计算由宽乘数α2 f1的叉积;0:75;0:5;0:25g和决议f224;192;160;128g。

图5显示之间的权衡ImageNet准确性和参数的数量为16模型由宽乘数α2 f1的叉积;0:75;0:5;0:25g和决议f224;192;160;128 g。

表8将完整的MobileNet与原始的GoogleNet[30]和VGG16[27]进行了比较。MobileNet几乎与VGG16一样精确，但它比VGG16小32倍，计算强度低27倍。它比GoogleNet更精确，但体积更小，计算量是后者的2.5倍。表9比较与宽度减少MobileNet乘数α= 0:5和降低分辨率160×160。简化的MobileNet比AlexNet[19]好4%，同时比AlexNet小45倍和少9:4倍的计算。在相同的尺寸和22倍的计算量下，它也比Squeezenet[12]好4%。

精读
实验3：不同宽度的MobileNet效果对比

   表6展示了不同width缩放因子α对准确率的影响



结论：  当α=0.25时才有显著下降。

实验4：不同分辨率的MobileNet效果对比

   表7展示了不同分辨率ρ对准确率的影响



结论：准确度随着分辨率变低而下降。

实验5：精度和MAdds、参数量的关系

   图4展示MAdds和精度的关系



结论：当模型越来越小时，精度可近似看成对数跳跃形式的。

图5展示了精度和参数量的关系



结论：  参数量高的精度对应也高。

实验6：和先进网络模型对比

表8比较了MobileNet与GoogleNet和VGG16的效果(大网络)



结论：

MobileNet和VGG16准确度相似，但是小了32倍，计算量少了27倍。
MobileNet比GoogleNet准确率更高，但是更小，计算量少了2.5倍。
表9比较了用width multiplier α=0.5和resolution160*160减少后的MobileNet



结论：

Reduced MobileNet比AlexNet准确度高4%，但是小了45倍，计算量少了9.4倍。
MobileNet比同等大小的SqueezeNet准确度高了4%，计算量少了22倍。
4.3 Fine Grained Recognition—细粒度图像分类
翻译
我们训练MobileNet在斯坦福狗数据集[17]上进行细粒度识别。我们扩展了[18]的方法，并从web上收集了比[18]更大但更嘈杂的训练集。我们使用带噪声的web数据预训练细粒度的犬类识别模型，然后在Stanford Dogs training set上对模型进行微调。Stanford Dogs test set的结果如表10所示。MobileNet几乎可以实现[18]的最先进的结果，大大减少了计算和大小。

精读
实验7：在Stanford Dogs数据集上训练MobileNet

表10用网络噪声数据来预训练一个细粒度狗的识别模型，在Stanford Dogs训练集上进行微调



结论： 和Inception v3对比，MobileNet在计算量和参数量降低一个数量级的同时几乎保持相同的精度。

4.4 Large Scale Geolocalizaton—以图搜地
翻译
行星[35]把确定照片拍摄地点的任务作为一个分类问题。该方法将地球划分成网格状的地理单元，这些单元作为目标类，并在数百万张带有地理标记的照片上训练一个卷积神经网络。PlaNet已经被证明能够成功地定位大量的照片，并且在处理相同任务方面表现得比Im2GPS更出色[6,7]。我们在相同的数据上使用MobileNet架构对PlaNet进行再培训。而基于Inception V3架构[31]的完整行星模型有5200万个参数和57.4亿个mult-add。MobileNet模型只有1300万个参数，通常是300万个参数用于车身，1000万个参数用于最后一层，58万个多附加参数。如表11所示，MobileNet版本虽然更紧凑，但与PlaNet相比，其性能仅略有下降。此外，它的表现仍然远远好于Im2GPS。

精读
实验8：基于以图搜地，对比不同网络结构进行对比

表11在相同的数据上使用MobileNet架构重新训练PlaNet，然后进行对比



结论：  MobileNet版本与PlaNet相比，虽然更紧凑，但性能仅略有下降。但它的性能仍然大大优于Im2GPS。

4.5 Face Attributes—人脸属性
翻译
MobileNet的另一个用例是压缩大型系统,有未知或深奥的培训程序。在人脸属性分类任务中,我们展示了MobileNet与蒸馏之间的协同关系,这是一种深度网络的知识传递技术。我们寻求通过7500万的参数和1600万的mult补充来减少一个大的面部属性分类器。分类器在一个与YFCC100M[32]相似的多属性数据集上进行了训练。我们使用MobileNet架构进行人脸属性分类器。通过训练分类器来模拟更大的模型2的输出,而不是地面真理标签,因此可以通过培训分类器来模拟大模型的输出,从而使从大型(以及潜在的无限)未标记的数据集进行培训。结合蒸馏训练的可扩展性和MobileNet的无偏性参数化,最终系统不仅不需要正规化(例如重量衰变和早期停止),而且还演示了增强的性能。从表12中可以看出,mobilenetbased的分类器对攻击性模型的收缩很有弹性:它在跨属性(指AP)中达到了类似的平均精度(指AP),但仅消耗1%的多值。

精读
实验9：人脸属性分类任务

表12演示了在MobileNet和蒸馏法之间的协同关系



结论： 基础MobileNet分类器具有积极的模型缩小的弹性：它在人脸属性上实现了一个相似的平均精度却只消耗了Mult-Adds的 1%。

4.6 Object Detection— 目标检测
翻译
MobileNet也可以作为一个有效的基础网络部署在现代目标检测系统。我们报告的结果，MobileNet训练的目标检测基于COCO数据基于最近的工作，赢得了2016年COCO挑战[10]。在表13中，MobileNet在fast - rcnn和SSD框架下与VGG和Inception V2进行了比较。在我们的实验中，SSD被评估为300输入分辨率(SSD 300)， fast - rcnn被评估为300和600输入分辨率(FasterRCNN 300, fast - rcnn 600)。快速rcnn模型对每个图像评估300个RPN建议框。模型在COCO训练+val上进行训练，排除8k minival图像，在minival上进行评估。对于这两个框架，MobileNet实现了与其他网络的可比较的结果，只有一小部分的计算复杂性和模型大小。

精读
实验10：COCO数据目标检测训练

表13演示了MobileNet与VGG和Faster-RCNN和SSD框架的比较



结论：  MobileNet和另外两个网络相比计算复杂度和模型大小都相对更小

4.7 Face Embeddings—人脸识别
翻译
FaceNet模型是目前最先进的人脸识别模型。它建立基于三重损耗的面嵌入。为了建立一个移动的FaceNet模型，我们使用蒸馏来最小化FaceNet和MobileNet输出在训练数据上的平方差异。非常小的MobileNet模型的结果可以在表14中找到。

精读
实验11：人脸识别检测

表14显示了基于MobileNet和蒸馏技术训练出结果

 

结论： 精读略有下降但计算量和参数大幅度减少

五、Conclusion—结论
翻译
提出了一种基于深度可分卷积的移动网络模型结构。我们研究了一些导致有效模型的重要设计决策。然后，我们演示了如何使用宽度倍增器和分辨率倍增器构建更小、更快的mobilenet，通过权衡一定的精度来减少大小和延迟。然后，我们比较了不同的移动网络和流行的模型，显示出更大的尺寸，速度和准确性的特点。最后，我们展示了MobileNet在广泛应用于各种任务时的有效性。作为帮助采用和探索MobileNets的下一步，我们计划在张量流中发布模型。

精读
（1）提出了一种基于深度可分离卷积构建了轻量级网络— MobileNet，并设置宽度乘子和分辨率乘子以调整网络大小达到在不同设备上适配的 目的。

（2）通过实验其在大幅度降低MAdds和参数量时，没有出现精度显著下降。

（3）在实验部分展示了MobileNet在分类、检测、人脸识别等各种图像任务上的效果，并且将MobileNets与其他先进的模型进行对比，凸显了 MobileNets良好的尺寸和性能。

## 🌟代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
 
# 定义DSC结构：DW+PW操作
def BottleneckV1(in_channels, out_channels, stride):
    # 深度可分卷积操作模块: DSC卷积 = DW卷积 + PW卷积
    return nn.Sequential(
        # dw卷积,也是RexNeXt中的组卷积，当分组个数等于输入通道数时，输出矩阵的通道输也变成了输入通道数时，组卷积就是dw卷积
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                  groups=in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace=True),
        # pw卷积，与普通的卷积一样，只是使用了1x1的卷积核
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )
 
 
# 定义MobileNetV1结构
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetV1, self).__init__()
 
        # torch.Size([1, 3, 224, 224])
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # torch.Size([1, 32, 112, 112])
 
        # 叠加的基本结构是： DW+PW(DW用来减小尺寸stride=2实现,PW用来增加通道out_channels增加实现)
        self.bottleneck = nn.Sequential(
            BottleneckV1(32, 64, stride=1),  # torch.Size([1, 64, 112, 112]), stride=1
            BottleneckV1(64, 128, stride=2),  # torch.Size([1, 128, 56, 56]), stride=2
            BottleneckV1(128, 128, stride=1),  # torch.Size([1, 128, 56, 56]), stride=1
            BottleneckV1(128, 256, stride=2),  # torch.Size([1, 256, 28, 28]), stride=2
            BottleneckV1(256, 256, stride=1),  # torch.Size([1, 256, 28, 28]), stride=1
            BottleneckV1(256, 512, stride=2),  # torch.Size([1, 512, 14, 14]), stride=2
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 512, stride=1),  # torch.Size([1, 512, 14, 14]), stride=1
            BottleneckV1(512, 1024, stride=2),  # torch.Size([1, 1024, 7, 7]), stride=2
            BottleneckV1(1024, 1024, stride=1),  # torch.Size([1, 1024, 7, 7]), stride=1
        )
 
        # torch.Size([1, 1024, 7, 7])
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)  # torch.Size([1, 1024, 1, 1])
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
 
        self.init_params()
 
    # 初始化操作
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        x = self.first_conv(x)  # torch.Size([1, 32, 112, 112])
        x = self.bottleneck(x)  # torch.Size([1, 1024, 7, 7])
        x = self.avg_pool(x)  # torch.Size([1, 1024, 1, 1])
        x = x.view(x.size(0), -1)  # torch.Size([1, 1024])
        x = self.dropout(x)
        x = self.linear(x)  # torch.Size([1, 5])
        out = self.softmax(x)  # 概率化
        return x
 
 
if __name__ == '__main__':
    net = MobileNetV1().cuda()
    summary(net, (3, 224, 224))
```

### 运行本项目

```bash
python
```

### 网络结构打印如下：

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 112, 112]             896
       BatchNorm2d-2         [-1, 32, 112, 112]              64
             ReLU6-3         [-1, 32, 112, 112]               0
            Conv2d-4         [-1, 32, 112, 112]             320
       BatchNorm2d-5         [-1, 32, 112, 112]              64
             ReLU6-6         [-1, 32, 112, 112]               0
            Conv2d-7         [-1, 64, 112, 112]           2,112
       BatchNorm2d-8         [-1, 64, 112, 112]             128
             ReLU6-9         [-1, 64, 112, 112]               0
           Conv2d-10           [-1, 64, 56, 56]             640
      BatchNorm2d-11           [-1, 64, 56, 56]             128
            ReLU6-12           [-1, 64, 56, 56]               0
           Conv2d-13          [-1, 128, 56, 56]           8,320
      BatchNorm2d-14          [-1, 128, 56, 56]             256
            ReLU6-15          [-1, 128, 56, 56]               0
           Conv2d-16          [-1, 128, 56, 56]           1,280
      BatchNorm2d-17          [-1, 128, 56, 56]             256
            ReLU6-18          [-1, 128, 56, 56]               0
           Conv2d-19          [-1, 128, 56, 56]          16,512
      BatchNorm2d-20          [-1, 128, 56, 56]             256
            ReLU6-21          [-1, 128, 56, 56]               0
           Conv2d-22          [-1, 128, 28, 28]           1,280
      BatchNorm2d-23          [-1, 128, 28, 28]             256
            ReLU6-24          [-1, 128, 28, 28]               0
           Conv2d-25          [-1, 256, 28, 28]          33,024
      BatchNorm2d-26          [-1, 256, 28, 28]             512
            ReLU6-27          [-1, 256, 28, 28]               0
           Conv2d-28          [-1, 256, 28, 28]           2,560
      BatchNorm2d-29          [-1, 256, 28, 28]             512
            ReLU6-30          [-1, 256, 28, 28]               0
           Conv2d-31          [-1, 256, 28, 28]          65,792
      BatchNorm2d-32          [-1, 256, 28, 28]             512
            ReLU6-33          [-1, 256, 28, 28]               0
           Conv2d-34          [-1, 256, 14, 14]           2,560
      BatchNorm2d-35          [-1, 256, 14, 14]             512
            ReLU6-36          [-1, 256, 14, 14]               0
           Conv2d-37          [-1, 512, 14, 14]         131,584
      BatchNorm2d-38          [-1, 512, 14, 14]           1,024
            ReLU6-39          [-1, 512, 14, 14]               0
           Conv2d-40          [-1, 512, 14, 14]           5,120
      BatchNorm2d-41          [-1, 512, 14, 14]           1,024
            ReLU6-42          [-1, 512, 14, 14]               0
           Conv2d-43          [-1, 512, 14, 14]         262,656
      BatchNorm2d-44          [-1, 512, 14, 14]           1,024
            ReLU6-45          [-1, 512, 14, 14]               0
           Conv2d-46          [-1, 512, 14, 14]           5,120
      BatchNorm2d-47          [-1, 512, 14, 14]           1,024
            ReLU6-48          [-1, 512, 14, 14]               0
           Conv2d-49          [-1, 512, 14, 14]         262,656
      BatchNorm2d-50          [-1, 512, 14, 14]           1,024
            ReLU6-51          [-1, 512, 14, 14]               0
           Conv2d-52          [-1, 512, 14, 14]           5,120
      BatchNorm2d-53          [-1, 512, 14, 14]           1,024
            ReLU6-54          [-1, 512, 14, 14]               0
           Conv2d-55          [-1, 512, 14, 14]         262,656
      BatchNorm2d-56          [-1, 512, 14, 14]           1,024
            ReLU6-57          [-1, 512, 14, 14]               0
           Conv2d-58          [-1, 512, 14, 14]           5,120
      BatchNorm2d-59          [-1, 512, 14, 14]           1,024
            ReLU6-60          [-1, 512, 14, 14]               0
           Conv2d-61          [-1, 512, 14, 14]         262,656
      BatchNorm2d-62          [-1, 512, 14, 14]           1,024
            ReLU6-63          [-1, 512, 14, 14]               0
           Conv2d-64          [-1, 512, 14, 14]           5,120
      BatchNorm2d-65          [-1, 512, 14, 14]           1,024
            ReLU6-66          [-1, 512, 14, 14]               0
           Conv2d-67          [-1, 512, 14, 14]         262,656
      BatchNorm2d-68          [-1, 512, 14, 14]           1,024
            ReLU6-69          [-1, 512, 14, 14]               0
           Conv2d-70            [-1, 512, 7, 7]           5,120
      BatchNorm2d-71            [-1, 512, 7, 7]           1,024
            ReLU6-72            [-1, 512, 7, 7]               0
           Conv2d-73           [-1, 1024, 7, 7]         525,312
      BatchNorm2d-74           [-1, 1024, 7, 7]           2,048
            ReLU6-75           [-1, 1024, 7, 7]               0
           Conv2d-76           [-1, 1024, 7, 7]          10,240
      BatchNorm2d-77           [-1, 1024, 7, 7]           2,048
            ReLU6-78           [-1, 1024, 7, 7]               0
           Conv2d-79           [-1, 1024, 7, 7]       1,049,600
      BatchNorm2d-80           [-1, 1024, 7, 7]           2,048
            ReLU6-81           [-1, 1024, 7, 7]               0
        AvgPool2d-82           [-1, 1024, 1, 1]               0
          Dropout-83                 [-1, 1024]               0
           Linear-84                    [-1, 5]           5,125
          Softmax-85                    [-1, 5]               0
================================================================
Total params: 3,223,045
Trainable params: 3,223,045
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 115.43
Params size (MB): 12.29
Estimated Total Size (MB): 128.30
----------------------------------------------------------------
```

### 强推的讲解视频

- [【精读AI论文】谷歌轻量化网络MobileNet V1（附MobileNetV1实时图像分类代码）](https://www.bilibili.com/video/BV1yE411p7L7)
- [7.1 MobileNet网络详解](https://www.bilibili.com/video/BV1yE411p7L7)

---

**版权声明**：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。

原文链接：https://blog.csdn.net/weixin_43334693/article/details/130719159