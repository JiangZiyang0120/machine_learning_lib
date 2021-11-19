#ifndef MACHINE_LEARNING_DIVISION_H
#define MACHINE_LEARNING_DIVISION_H
#include <vector>
#include <set>


/*
 * 将数据集划分为训练集和测试集
 * 采用分层抽样形式
 * 通过tuple返回训练集dataSet,targetSet，测试集dataSet,targetSet
 */
template<typename Data, typename Target>
auto train_test_split(std::vector<Data> &, std::vector<Target> &, double);


/*
 * 留出法(hold-out)
 * 将数据集划分为互斥的两个集合分别作为训练集和数据集
 * 根据学习模型学习后返回精度
 */
template<typename Data, typename Target, typename model>
double hold_out(std::vector<Data> &, std::vector<Target> &, double,
                model &, double (*)(std::vector<Data>, std::vector<Target>, model));


/*
 * k折交叉验证法(k-fold-cross-validation)
 * 分层抽样将数据分为k份，依次将一份数据作为测试集而将其他作为训练集
 * 根据学习模型进行验证后返回k个结果的均值
 */
template<typename Data, typename Target, typename model>
double k_fold_cross_validation(std::vector<Data> &, std::vector<Target> &, size_t,
                               model &, double (*)(std::vector<Data>, std::vector<Target>, model));

/*
 * 自助法(bootstrapping)
 * 对于整体样本D，通过m次采样获取训练集D' 与测试集D\D'
 * 根据该训练集训练并返回精确度
 */
template<typename Data, typename Target, typename model>
double bootstrapping(std::vector<Data> &, std::vector<Target> &, size_t,
                     model &, double (*)(std::vector<Data>, std::vector<Target>, model));


/*
 *
 */

/*
 * 本函数将数据集和测试集等量分为k份
 * 需注意，由于方法选取问题，不同份之间样本数并不完全相同
 * 对于某些样本而言不同样本数差异可能多达 Target Type 的数量
 */
template<typename Data, typename Target>
auto k_times_division(std::vector<Data> &dataSet, std::vector<Target> &targetSet, size_t k) {
    std::set<Target> targetTypeSet;
    std::vector<std::vector<Data>> dataSubsectionSet;
    std::vector<std::vector<Target>> targetSubsectionSet;
    //提前分配空间，避免resize带来的消耗
    dataSubsectionSet.reserve(k);
    targetSubsectionSet.reserve(k);
    //对数据集进行分段操作
    for (auto iter = targetSet.cbegin(); iter != targetSet.cend(); ++iter) {
        targetTypeSet.insert(*iter);
    }
    std::map<Target, std::vector<Data>> subsectionSet;
    for (auto iter = targetTypeSet.cbegin(); iter != targetTypeSet.cend(); ++iter) {
        subsectionSet[*iter] = std::vector<Data>();
    }
    for (size_t i = 0; i != dataSet.size(); ++i) {
        subsectionSet[targetSet[i]].push_back(dataSet[i]);
    }
    //随机排序并进行划分
    for (auto iter = subsectionSet.begin(); iter != subsectionSet.end(); ++iter) {
        Fisher_Yates(iter->second); //进行随机排序
        auto &eachTargetSet = iter->second;
        size_t length = eachTargetSet.size();
        for (size_t i = 0; i != k; ++i) {
            dataSubsectionSet.push_back(std::vector<Data>());
            targetSubsectionSet.push_back(std::vector<Target>());
            dataSubsectionSet[i].insert(
                    dataSubsectionSet[i].end(),
                    eachTargetSet.begin() + round(static_cast<double >(i * length) / static_cast<double >(k)),
                    eachTargetSet.begin() + round(static_cast<double >((i + 1) * length) / static_cast<double >(k))
            );
            size_t eachSizeOfSet = round(static_cast<double >((i + 1) * length) / static_cast<double >(k) -
                                         static_cast<double >(i * length) / static_cast<double >(k));
            targetSubsectionSet[i].reserve(eachSizeOfSet);
            for (size_t j = 0; j != eachSizeOfSet; ++j) {
                targetSubsectionSet[i].push_back(iter->first);
            }
        }
    }
    return std::tuple(dataSubsectionSet, targetSubsectionSet, k);
}


template<typename Data, typename Target>
auto train_test_split(std::vector<Data> &dataSet, std::vector<Target> &targetSet, double trainSetPropotion) {
    std::set<Target> targetTypeSet;
    std::vector<Data> trainDataSet, testDataSet;
    std::vector<Target> trainTargetSet, testTargetSet;
    //将数据集进行分段操作
    for (auto iter = targetSet.cbegin(); iter != targetSet.cend(); ++iter) {
        targetTypeSet.insert(*iter);
    }
    std::map<Target, std::vector<Data>> subsectionSet;
    for (auto iter = targetTypeSet.cbegin(); iter != targetTypeSet.cend(); ++iter) {
        subsectionSet[*iter] = std::vector<Data>();
    }
    for (size_t i = 0; i != dataSet.size(); ++i) {
        subsectionSet[targetSet[i]].push_back(dataSet[i]);
    }
    //随机排序并进行划分
    for (auto iter = subsectionSet.begin(); iter != subsectionSet.end(); ++iter) {
        Fisher_Yates(iter->second); //进行随机排序
        size_t trainSize = round(trainSetPropotion * iter->second.size());
        trainDataSet.insert(
                trainDataSet.end(), iter->second.begin(), iter->second.begin() + trainSize);
        testDataSet.insert(
                testDataSet.end(), iter->second.begin() + trainSize, iter->second.end());
        for (size_t i = 0; i != trainSize; ++i) trainTargetSet.push_back(iter->first);
        for (size_t i = trainSize; i != iter->second.size(); ++i) testTargetSet.push_back(iter->first);
    }
    return std::tuple(trainDataSet, trainTargetSet, testDataSet, testTargetSet);
}


template<typename Data, typename Target, typename model>
double hold_out(std::vector<Data> &dataSet, std::vector<Target> &targetSet, double trainSetPropotion,
                model &machineLearningModel,
                double (*machineLearningInspection)(std::vector<Data>, std::vector<Target>, model)) {
    auto[trainDataSet, trainTargetSet, testDataSet, testTargetSet] =
    train_test_split(dataSet, targetSet, trainSetPropotion);
    model thisModel = machineLearningModel.modelConsrtruction(trainDataSet, trainTargetSet);
    return machineLearningInspection(testDataSet, testTargetSet, thisModel);
}


template<typename Data, typename Target, typename model>
double k_fold_cross_validation(std::vector<Data> &dataSet, std::vector<Target> &targetSet, size_t k,
                               model &machineLearningModel,
                               double (*machineLearningInspection)(std::vector<Data>, std::vector<Target>, model)) {
    //进行k等分
    auto[dataSubsectionSet, targetSubsectionSet, thisK] = k_times_division(dataSet, targetSet, k);
    //根据模型返回结果
    double sum = 0;
    for (size_t ki = 0; ki != thisK; ++ki) {
        size_t testLength = dataSubsectionSet[ki].size();
        //设置训练集测试集
        std::vector<Data> trainDataSet, testDataSet;
        std::vector<Target> trainTargetSet, testTargetSet;
        testDataSet.reserve(testLength);
        testTargetSet.reserve(testLength);
        for (size_t i = 0; i != thisK; ++i) {
            if (i != ki) {
                trainDataSet.insert(trainDataSet.end(),
                                    dataSubsectionSet[i].begin(),
                                    dataSubsectionSet[i].end());
                trainTargetSet.insert(trainTargetSet.end(),
                                      targetSubsectionSet[i].begin(),
                                      targetSubsectionSet[i].end());
            } else {
                testDataSet.insert(testDataSet.end(),
                                   dataSubsectionSet[i].begin(),
                                   dataSubsectionSet[i].end());
                testTargetSet.insert(testTargetSet.end(),
                                     targetSubsectionSet[i].begin(),
                                     targetSubsectionSet[i].end());
            }
        }
        //训练模型并进行检测
        auto thisModel = machineLearningModel.modelConsrtruction(trainDataSet, trainTargetSet);
        sum += machineLearningInspection(testDataSet, testTargetSet, thisModel);
    }
    return sum / thisK;
}


template<typename Data, typename Target, typename model>
double bootstrapping(std::vector<Data> &dataSet, std::vector<Target> &targetSet, size_t m,
                     model &machineLearningModel,
                     double (*machineLearningInspection)(std::vector<Data>, std::vector<Target>, model)) {
    size_t dataSetSize = dataSet.size();
    std::vector<Data> trainDataSet, testDataSet;
    std::vector<Target> trainTargetSet, testTargetSet;
    std::set<Data> samplingDataSet; //需注意，这里没有考虑Data相同而Target不同的情况
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<size_t> dist(0, dataSetSize - 1);
    //执行抽样工作
    for (size_t i = 0; i != m; ++i) {
        size_t index = dist(mt);
        samplingDataSet.insert(dataSet[index]);
    }
    //根据上面的抽样将数据划分为训练集和测试集
    for (size_t i = 0; i != dataSetSize; ++i) {
        if (samplingDataSet.find(dataSet[i])) {
            trainDataSet.push_back(dataSet[i]);
            trainTargetSet.push_back(targetSet[i]);
        } else {
            testDataSet.push_back(dataSet[i]);
            testTargetSet.push_back(targetSet[i]);
        }
    }
    //训练模型并返回精确度
    model thisModel = machineLearningModel.modelConsrtruction(trainDataSet, trainTargetSet);
    return machineLearningInspection(testDataSet, testTargetSet, thisModel);
}

#endif //MACHINE_LEARNING_DIVISION_H
