
#include "dataset.hpp"
#include "NN.hpp"
#include "../lib/matrix.h"
#include <vector>
#include <iostream>

void debug(Example e) {
    static std::string shades = " .:-=+*#%@";
    for (unsigned int i = 0; i < 28 * 28; i++) {
        if (i % 28 == 0) printf("\n");
        printf("%c", shades[e.data[i] / 30]);
    }
    printf("\nLabel: %d\n", e.label);
}

std::vector<double> load_matrix(Example& e) {
    std::vector<double> result(e.data, e.data + 28 * 28);
    return result;
}

const double calculate_accuracy(const Matrix<unsigned char>& images, const Matrix<unsigned char>& labels, NeuralNetwork n) {
  unsigned int correct = 0;
  for (unsigned int i = 0; i < images.rows(); ++i) {
    Example e;
    for (int j = 0; j < 28*28; ++j) {
        e.data[j] = images[i][j];
    }
    e.label = labels[i][0];
    unsigned int guess = n.compute(e);
    if (guess == (unsigned int)e.label) correct++;
  }
  const double accuracy = (double)correct/images.rows();

  return accuracy;
}

void tests(int count){
    double sum_train = 0;
    double sum_test = 0;
    for(int i =0; i <= count; i++){
        Matrix<unsigned char> images_train(0, 0);
   
        Matrix<unsigned char> labels_train(0, 0);
   
        load_dataset(images_train, labels_train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");

   
        Matrix<unsigned char> images_test(0, 0);
        Matrix<unsigned char> labels_test(0, 0);
        load_dataset(images_test, labels_test, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

        NeuralNetwork n;

        const unsigned int num_iterations = 5;
        n.train(num_iterations, images_train, labels_train);

        const double accuracy_train = calculate_accuracy(images_train, labels_train, n);
        const double accuracy_test = calculate_accuracy(images_test, labels_test, n);
        
        sum_train += accuracy_train;
        sum_test += accuracy_test;
        
        printf("Accuracy on training data: %f\n", accuracy_train);
        printf("Accuracy on test data: %f\n", accuracy_test);
    
    };
    printf(" ----------------------------------------\n Average accuracy of train data: %f\n", sum_train/count);
    printf(" Average accuracy of test data: %f\n", sum_test/count);
}

#ifdef TESTS
#include "gtest/gtest.h"

double max(double a, double b) {
    if(a >= b)
       return a;
    else
       return b;
}

std::vector<double> ReLU(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = max(x[i], 0);
    return result;
}

TEST(FunctionTesting, testMax1){
    EXPECT_NEAR(max(0.37, 0.29),0.37,1e-6);
    EXPECT_NEAR(max(-0.52, -0.92),-0.52,1e-6);
    EXPECT_NEAR(max(0, 0.83),0.83,1e-6);
}

TEST(FunctionTesting, testMax2){
    EXPECT_NEAR(max(0.1, 0.0 * 0.1),0.1,1e-6);
    EXPECT_NEAR(max(-0.1, 0.0 * (-0.1)),0.0,1e-6);
    EXPECT_NEAR(max(0, 0.0 * 0),0.0,1e-6);
}

TEST(FunctionTesting, testReLUPos){
    std::vector<double> x1 = {0.13, 0.23, 0.33, 0.43, 0.53};
    std::vector<double> right_x1 = {0.13, 0.23, 0.33, 0.43, 0.53};
    ASSERT_EQ(ReLU(x1),right_x1);
}

TEST(FunctionTesting, testReLUMix){
    std::vector<double> x2 = {0.05, -0.45, -0.24, 0.01, -0.99};
    std::vector<double> right_x2 = {0.05, 0.0, 0.0, 0.01, 0.0};
    ASSERT_EQ(ReLU(x2),right_x2);
}

TEST(FunctionTesting, testReLUNeg){
    std::vector<double> x3 = {-0.75, -0.93, -0.38, -0.02, -0.63};
    std::vector<double> right_x3 = {0.0, 0.0, 0.0, 0.0, 0.0};
    ASSERT_EQ(ReLU(x3),right_x3);
}
#endif

int main() {
   tests(6);
   #ifdef TESTS
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    #endif
      Matrix<unsigned char> images_train(0, 0);
    Matrix<unsigned char> labels_train(0, 0);
    load_dataset(images_train, labels_train, "data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    Matrix<unsigned char> images_test(0, 0);
    Matrix<unsigned char> labels_test(0, 0);
    load_dataset(images_test, labels_test, "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    
    NeuralNetwork n;
   
    const unsigned int num_iterations = 5;
    n.train(num_iterations, images_train, labels_train);
   
    const double accuracy_train = calculate_accuracy(images_train, labels_train, n);
    const double accuracy_test = calculate_accuracy(images_test, labels_test, n);
    
    printf("Accuracy on training data: %f\n", accuracy_train);
    printf("Accuracy on test data: %f\n", accuracy_test
   
        return 0;
}
