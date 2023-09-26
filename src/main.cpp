
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
    return a > 0 ? a : a / sqrt(1 + alpha * a * a);
}

std::vector<double> isrlu(const std::vector<double>& x, double alpha) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = isrlu(x[i], alpha);
    return result;
}

TEST(FunctionTesting, testIsrlu1) {
    EXPECT_NEAR(isrlu(-2.5, 0.3), -0.521612, 1e-5);
    EXPECT_NEAR(isrlu(1.8, 0.3), 1.8, 1e-5);
    EXPECT_NEAR(isrlu(-0.7, 0.3), -0.443049, 1e-5);
}

TEST(FunctionTesting, testIsrlu2) {
    EXPECT_NEAR(isrlu(0.2, 0.5), 0.2, 1e-5);
    EXPECT_NEAR(isrlu(0, 0.5), 0, 1e-5);
    EXPECT_NEAR(isrlu(-0.9, 0.5), -0.575646, 1e-5);
}

TEST(FunctionTesting, testIsrluPos) {
    std::vector<double> x1 = {0.56, 0.99, 1.8, 2.1, 0.53};
    std::vector<double> right_x1 = {0.56, 0.99, 1.8, 2.1, 0.53};

    std::vector<double> result = isrlu(x1, 0.7);

    for (unsigned int i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], right_x1[i], 1e-5);
    }
}

TEST(FunctionTesting, testIsrluMix) {
    std::vector<double> x2 = {0.5, -0.4, -0.33, 0.1, -0.92};
    std::vector<double> right_x2 = {0.5, -0.253568, -0.217129, 0.0866035, -0.730297};

    std::vector<double> result = isrlu(x2, 0.4);

    for (unsigned int i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], right_x2[i], 1e-5);
    }
}

TEST(FunctionTesting, testIsrluNeg) {
    std::vector<double> x3 = {-0.75, -0.93, -0.38, -0.02, -0.63};
    std::vector<double> right_x3 = {-0.610124, -0.698148, -0.408248, -0.0200004, -0.541587};

    std::vector<double> result = isrlu(x3, 0.5);

    for (unsigned int i = 0; i < result.size(); i++) {
        EXPECT_NEAR(result[i], right_x3[i], 1e-5);
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
