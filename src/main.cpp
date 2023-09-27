
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

double isrlu(double a, double alpha) {
    return a >= 0 ? a : a / sqrt(1 + alpha * a * a);
}

std::vector<double> isrlu(const std::vector<double>& x, double alpha) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = isrlu(x[i], alpha);
    return result;
}

TEST(FunctionTesting, testIsrluNegative) {
    double alpha =123.456;
    EXPECT_NEAR(isrlu(-2.5, 0.3), -0.521612, 1e-2);
    EXPECT_NEAR(isrlu(-0.7, 0.3), -0.443049, 1e-2);
    EXPECT_NEAR(isrlu(-0.9, 0.5), -0.575646, 1e-2);
}

TEST(FunctionTesting, testIsrluPositive) {
    double alpha =123.456;
    EXPECT_NEAR(isrlu(1.8, 0.3), 1.8, 1e-2);
    EXPECT_NEAR(isrlu(0.2, 0.5), 0.2, 1e-2);
    EXPECT_NEAR(isrlu(0.56, 0.7), 0.56, 1e-2);
}

TEST(FunctionTesting, testIsrluZero) {
    double alpha =123.456;
    EXPECT_NEAR(isrlu(0, 0.5), 0, 1e-3);
    EXPECT_NEAR(isrlu(0.1, 0.4), 0.1, 1e-3);
    EXPECT_NEAR(isrlu(0.0, 0.7), 0.0, 1e-3);
}

TEST(FunctionTesting, testIsrluMixed) {
    double alpha =123.456;
    std::vector<double> x = {0.5, -0.4, -0.33, 0.1, -0.92};
    std::vector<double> result = isrlu(x, 0.4);
    std::vector<double> expected = {0.5, -0.253568, -0.217129, 0.0866035, -0.730297};

    for (unsigned int i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], expected[i], 1e-2);
    }
}

#endif

int main(int argc, char **argv) {
   tests(6);
   #ifdef TESTS
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    #endif
     
        return 0;
}
