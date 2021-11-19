#ifndef MACHINE_LEARNING_UTIL_ALGORITHM_H
#define MACHINE_LEARNING_UTIL_ALGORITHM_H
#include <random>


/*
 * 通过Fisher-Yates算法进行抽样
 */
template <typename T>
void Fisher_Yates(T &);


/*
 * 多文件编程范型要求非专用模板对其翻译单元可见
 * 因此该头文件中必须包含该范型算法的实现
 * 本例中采用包含源文件的方式
 * 需注意，为免重复包含，源文件中不可include头文件
 */
template <typename T>
void Fisher_Yates(T &array){
    std::random_device rd;
    std::mt19937 mt(rd());
    for(size_t i = array.size()-1; i != 0; --i){
        std::uniform_int_distribution<size_t> dist(0,i);
        size_t k = dist(mt);
        auto temp = array[i];
        array[i] = array[k];
        array[k] = temp;
    }
}

#endif //MACHINE_LEARNING_UTIL_ALGORITHM_H