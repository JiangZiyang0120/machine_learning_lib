#ifndef MAIN_CPP_LINEAR_MODEL_H
#define MAIN_CPP_LINEAR_MODEL_H

#include "ml_base_model.h"
#include "matrix.h"

/*
 * 线性回归模型
 */
template<typename Data = double, typename Target = double, typename Coefficient = double>
class LinearModel : public BaseModel<Data, Target> {
public:
    /*
     * 构造函数
     */
    LinearModel();

    /*
     * 重载基类中的构建函数
     * 返回LinearModel的左值引用
     */
    LinearModel &model_construction(Matrix<Data> &, Matrix<Target> &) override;

    /*
     * 重载基类中的预测函数
     * 根据已学习得的模型返回预测值
     */
    Matrix<Target> &return_predict_target(Matrix<Data> &) override;

    /*
     * 操作符operator()重载
     * 根据传入的vector或matrix返回对应的1×n系数矩阵
     */
    Matrix<Coefficient> &operator()(Matrix<Data> &, Matrix<Target> &);

private:
    /*
     * 线性模型的各项系数
     * 各系数按先后顺序依次为Beta0（常数项）,Beta_1,...,Beta_n
     */
    std::vector<Coefficient> Beta;
};

/*
 * 对数几率回归
 */
template<typename Data = double, typename Target = double, typename Coefficient = double>
class LogitModel : public BaseModel<Data, Target> {
public:
    /*
     * 构造函数
     */
    LogitModel();

    /*
     * 重载基类中的构建函数
     * 返回LinearModel的左值引用
     */
    LogitModel &model_construction(Matrix<Data> &, Matrix<Target> &) override;

    /*
     * 重载基类中的预测函数
     * 根据已学习得的模型返回预测值
     */
    Matrix<Target> &return_result_target(Matrix<Data> &) override;

private:
    /*
     * 线性模型的各项系数
     * 各系数按先后顺序依次为Beta0（常数项）,Beta1,...,Betan
     */
    std::vector<Coefficient> Beta;

    /*
     * 分别为牛顿法和梯度下降法
     * 对系数矩阵进行凸优化，求其数值解
     */
    void Newton_method();

    void gradient_descent_method();
};


#endif //MAIN_CPP_LINEAR_MODEL_H
