import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 源域和目标域的数据 (简单模拟)
source_data = torch.randn(100, 10)
source_labels = torch.randint(0, 2, (100,))  # 二分类
target_data = torch.randn(100, 10)

# 数据加载器
source_loader = DataLoader(TensorDataset(source_data, source_labels), batch_size=16, shuffle=True)
target_loader = DataLoader(TensorDataset(target_data), batch_size=16, shuffle=True)

# 定义简单的特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer = nn.Linear(10, 50)

    def forward(self, x):
        return self.layer(x)

# 定义分类器
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(50, 2)

    def forward(self, x):
        return self.layer(x)

# 定义域判别器
class DomainDiscriminator(nn.Module):
    def __init__(self):
        super(DomainDiscriminator, self).__init__()
        self.layer = nn.Linear(50, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

# 初始化模型
feature_extractor = FeatureExtractor()
classifier = Classifier()
domain_discriminator = DomainDiscriminator()

# 损失函数和优化器
classification_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCELoss()
optimizer_feature = optim.Adam(feature_extractor.parameters(), lr=0.001)
optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.001)
optimizer_domain = optim.Adam(domain_discriminator.parameters(), lr=0.001)

# 训练域适应
for epoch in range(10):  # 训练 10 个 epoch
    feature_extractor.train()
    classifier.train()
    domain_discriminator.train()

    # 源域分类训练
    for source_batch, source_label in source_loader:
        source_feature = feature_extractor(source_batch)
        class_prediction = classifier(source_feature)
        loss_class = classification_criterion(class_prediction, source_label)

        # 更新特征提取器和分类器
        optimizer_feature.zero_grad()
        optimizer_classifier.zero_grad()
        loss_class.backward()
        optimizer_feature.step()
        optimizer_classifier.step()

    # 对抗训练进行域对齐
    for source_batch, _ in source_loader:
        for target_batch, _ in target_loader:
            # 源域特征
            source_feature = feature_extractor(source_batch)
            source_domain = domain_discriminator(source_feature)

            # 目标域特征
            target_feature = feature_extractor(target_batch)
            target_domain = domain_discriminator(target_feature)

            # 域标签 (1 表示源域, 0 表示目标域)
            domain_labels_source = torch.ones(source_domain.size(0), 1)
            domain_labels_target = torch.zeros(target_domain.size(0), 1)

            # 计算域损失
            loss_domain_source = domain_criterion(source_domain, domain_labels_source)
            loss_domain_target = domain_criterion(target_domain, domain_labels_target)
            loss_domain = loss_domain_source + loss_domain_target

            # 更新域判别器
            optimizer_domain.zero_grad()
            loss_domain.backward()
            optimizer_domain.step()

print("Domain adaptation training completed.")
