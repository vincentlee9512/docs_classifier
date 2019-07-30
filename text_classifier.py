"""
Author: 李文轩

中文文档分类 - 多项式贝叶斯分类算法实践

数据文档：https://github.com/cystanford/text_classification

项目阶段：

1. 文档输入

2. 准备阶段
    1. 对文档分词
    2. 加载停用词
    3. 计算单词权重

3. 分词阶段
    1. 生成分类器
    2. 分类器做预测
    3. 计算正确率


"""


import os
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def cut_words(file_path):
    """
    对文本进行切词

    :param file_path: txt文本路径
    :return: 用空格分词对字符串
    """
    res = ''

    # gb18030 中文编码
    text = open(file_path, 'r', encoding='gb18030').read()
    textcut = jieba.cut(text)
    for word in textcut:
        res += word + ' '
    return res

def loadfile(file_dir, label):
    """
    将路径下下的所有文件加载
    :param file_dir: 保存 txt 文件目录
    :param label: 文档标签
    :return: 分词后的文档列表和标签
    """
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = file_dir + '/' + file
        words_list.append(cut_words(file_path))
        labels_list.append(label)

    return words_list, labels_list


def main():
    """
    这个项目的 main 函数
    """


    """
    2.1 对文档分词：和加载文件一起。细节在 cut_words 函数中
    """
    # 加载训练数据
    train_words_list1, train_labels1 = loadfile('./train/体育', '体育')
    train_words_list2, train_labels2 = loadfile('./train/女性', '女性')
    train_words_list3, train_labels3 = loadfile('./train/文学', '文学')
    train_words_list4, train_labels4 = loadfile('./train/校园', '校园')

    train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
    train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4


    # 加载测试数据
    test_words_list1, test_labels1 = loadfile('./test/体育', '体育')
    test_words_list2, test_labels2 = loadfile('./test/女性', '女性')
    test_words_list3, test_labels3 = loadfile('./test/文学', '文学')
    test_words_list4, test_labels4 = loadfile('./test/校园', '校园')

    test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
    test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

    """
    2.2 加载停用词
    """
    stop_words = open('./stop/stopword.txt', 'r', encoding='utf-8').read()
    stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff 处理
    stop_words = stop_words.split('\n')


    """
    2.3 计算单词权重
    """
    tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)
    train_features = tf.fit_transform(train_words_list)
    test_features = tf.transform(test_words_list)

    """
    3.1 生成 朴素贝叶斯分类器
    """
    clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)

    """
    3.2 使用分类器做预测
    """
    predicted_labels = clf.predict(test_features)

    """
    3.3 计算准确率
    """
    curr_percentage = metrics.accuracy_score(test_labels, predicted_labels)
    print('准确率为：{}'.format(curr_percentage))


if __name__ == '__main__':
    main()
