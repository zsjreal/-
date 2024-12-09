from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 假设这是我们的训练数据
documents = [
    "I love programming in Python",
    "Python is great for data science",
    "I hate bugs in my code",
    "Debugging is annoying",
]
labels = [1, 1, 0, 0]  # 1: positive sentiment, 0: negative sentiment

# 简单的知识图谱模拟，提供情感权重
concept_knowledge = {
    "love": 1.0,
    "great": 1.0,
    "hate": -1.0,
    "annoying": -1.0,
}

# 知识注入函数：根据文档生成新的特征
def inject_knowledge(doc):
    words = doc.split()
    knowledge_score = sum(concept_knowledge.get(word.lower(), 0) for word in words)
    return knowledge_score

# 生成新的特征
knowledge_features = [inject_knowledge(doc) for doc in documents]

# 创建文本特征和知识特征的组合
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(documents)

# 将知识特征加入文本特征中
import scipy.sparse
X_combined = scipy.sparse.hstack((X_text, [[score] for score in knowledge_features]))

# 使用分类器
classifier = LogisticRegression()
classifier.fit(X_combined, labels)

# 测试
test_doc = "I love debugging Python"
test_text = vectorizer.transform([test_doc])
test_knowledge = [[inject_knowledge(test_doc)]]
test_combined = scipy.sparse.hstack((test_text, test_knowledge))

# 输出结果
print("Prediction:", classifier.predict(test_combined))  # 预测情感标签
