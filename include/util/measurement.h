#ifndef MACHINE_LEARNING_MEASUREMENT_H
#define MACHINE_LEARNING_MEASUREMENT_H

#include <vector>

/*
 * 学习模型的错误率
 * 适用于分类模型
 */
template <typename Data, typename Target, typename Model>
double err_rate(std::vector<Data> &, std::vector<Target> &, Model &);


/*
 * 学习模型的精确度
 */
template <typename Data, typename Target, typename Model>
double accuracy(std::vector<Data> &, std::vector<Target> &, Model &);

/*
 * 学习模型的查准率
 * 其定义为P = TP/(TP + FP)
 */
template <typename Data, typename Target, typename Model>
double precision(std::vector<Data> &, std::vector<Target> &, const Target &,
                 Model &);

/*
 * 学习模型的查全率
 * 其定义为P = TP/(TP + FN)
 */
template <typename Data, typename Target, typename Model>
double recall(std::vector<Data> &, std::vector<Target> &, const Target &,
                 Model &);


/*
 * 学习模型的F1度量
 * F1度量是基于查准率和查全率的调和平均（harmonic mean）
 * 其定义为1/F1 = (1/2)*(1/P + 1/R)
 */
template <typename Data, typename Target, typename Model>
double F1_measurement(std::vector<Data> &, std::vector<Target> &, const Target &,
              Model &);

/*
 * 学习模型的F1度量
 * 为了减小计算开销，该接口提供给已经计算出查准率和查全率的用户
 * 参数分别为查准量P(precision)和查全率R(recall)
 */
double F1_measurement(double ,double );


/*
 * 学习模型的加权调和平均
 * Beta > 0度量了查全率和查准率的相对重要性
 * Beta > 1时查全率影响更大，Beta < 1使查准率影响更大
 * Beta = 1时退化为F1 measurement
 */
template <typename Data, typename Target, typename Model>
double FBeta_measurement(std::vector<Data> &, std::vector<Target> &, const Target &,
                      Model &, double Beta);


/*
 * n个
 */

template<typename Data, typename Target, typename Model>
double err_rate(std::vector<Data> &testDataSet, std::vector<Target> &testTargetSet,
                Model &machineLearningModel) {
    auto testTargetResultSet = machineLearningModel.returnResultTarget(testDataSet);
    size_t targetSetSize = testTargetSet.size();
    size_t errNum = 0;
    for (size_t i = 0; i != targetSetSize; ++i) {
        if (testTargetResultSet[i] != testTargetSet[i]) ++errNum;
    }
    return static_cast<double>(errNum) / static_cast<double>(targetSetSize);
}


template<typename Data, typename Target, typename Model>
double accuracy(std::vector<Data> &testDataSet, std::vector<Target> &testTargetSet,
                Model &machineLearningModel) {
    return 1 - err_rate(testDataSet, testTargetSet, machineLearningModel);
}


template<typename Data, typename Target, typename Model>
double precision(std::vector<Data> &testDataSet, std::vector<Target> &testTargetSet,
                 const Target &trueTarget, Model &machineLearningModel) {
    size_t precisionTrue = 0, TP = 0;
    auto testTargetResultSet = machineLearningModel.returnResultTarget(testDataSet);
    for (size_t i = 0; i != testTargetResultSet.size(); ++i) {
        if (trueTarget == testTargetResultSet[i]) {
            ++precisionTrue;
            if (trueTarget == testTargetSet[i]) ++TP;
        }
    }
    return static_cast<double>(TP) / static_cast<double>(precisionTrue);
}


template<typename Data, typename Target, typename Model>
double recall(std::vector<Data> &testDataSet, std::vector<Target> &testTargetSet,
              const Target &trueTarget, Model &machineLearningModel) {
    size_t realTrue = 0, TP = 0;
    auto testTargetResultSet = machineLearningModel.returnResultTarget(testDataSet);
    for (size_t i = 0; i != testTargetResultSet.size(); ++i) {
        if (trueTarget == testTargetSet[i]) {
            ++realTrue;
            if (trueTarget == testTargetResultSet[i]) ++TP;
        }
    }
    return static_cast<double>(TP) / static_cast<double>(realTrue);
}


template<typename Data, typename Target, typename Model>
double F1_measurement(std::vector<Data> &testDataSet, std::vector<Target> &testTargetSet,
                      const Target &trueTarget, Model &machineLearningModel) {
    size_t TN = 0, TP = 0;
    auto testTargetResultSet = machineLearningModel.returnResultTarget(testDataSet);
    size_t sumInstance = testTargetResultSet.size();
    for (size_t i = 0; i != testTargetResultSet.size(); ++i) {
        if (testTargetResultSet[i] == testTargetSet[i]) {
            if (trueTarget == testTargetSet[i]) ++TP;
            else ++TN;
        }
    }
    return static_cast<double>(2 * TP) / static_cast<double>(sumInstance + TP - TN);
}


double F1_measurement(double P,double R){
    return (1/P + 1/R)/2;
}


template<typename Data, typename Target, typename Model>
double FBeta_measurement(std::vector<Data> &testDataSet, std::vector<Target> &testTargetSet,
                         const Target &trueTarget, Model &machineLearningModel, double Beta) {
    double P = precision(testDataSet, testTargetSet, trueTarget, machineLearningModel),
            R = recall(testDataSet, testTargetSet, trueTarget, machineLearningModel);
    return ((1 + Beta * Beta) * P * R) / (Beta * Beta * P + R);
}

#endif //MACHINE_LEARNING_MEASUREMENT_H
