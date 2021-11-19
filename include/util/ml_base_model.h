#ifndef MATRIX_H_ML_BASE_MODEL_H
#define MATRIX_H_ML_BASE_MODEL_H

#include "matrix.h"

/*
 *
 */
template <typename Data, typename Target>
class BaseModel{
public:
    /*
     * 构建函数，读入训练集的数据集和目标集
     * 根据derived class定义的方法训练模型
     * 返回BaseModel的引用。在编程中可以动态引用至derived class
     */
   virtual BaseModel &model_construction(Matrix<Data> &, Matrix <Target> &) = 0;
   /*
    * 根据训练集返回预测目标值的矩阵
    * 需返回n×1矩阵
    */
   virtual Matrix<Target> &return_predict_target(Matrix<Data> &) = 0;
};

#endif //MATRIX_H_ML_BASE_MODEL_H
