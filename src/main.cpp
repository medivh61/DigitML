
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
#include <gtest/gtest.h>

NeuralNetwork n;

std::vector<double> isrlu(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = x[i] >= 0 ? x[i] : x[i] / sqrt(1 + alpha * pow(x[i], 2));
    return result;
}

TEST(FunctionTesting, test_isrlu_zero) {
  std::vector<double> t1 = {0.0, 0.0, 0.0};
  std::vector<double> t2 = {0.0, 0.0, 0.0};
  double alpha =123.456; // Add this var in all tests
  EXPECT_EQ(n.isrlu(t1, alpha), t2);
}

TEST(FunctionTesting, test_isrlu_positive) {
  std::vector<double> t1 = {0.5, 1.0, 2.0};
  std::vector<double> t2 = {0.5, 1.0, 2.0};
  double alpha = 123.456;
  EXPECT_EQ(n.isrlu(t1), t2);
}

TEST(FunctionTesting, test_isrlu_negative) {
  std::vector<double> t1 = {-0.5, -1.0, -2.0};
  std::vector<double> t2 = {-0.5 / (1.0 + 0.5), -1.0 / (1.0 + 1.0), -2.0 / (1.0 + 2.0)};
  double alpha = 123.456;
  EXPECT_EQ(n.isrlu(t1), t2);
}

TEST(FunctionTesting, test_isrlu_mixed) {
  std::vector<double> t1 = {1.5, -2.0, 0.0};
  std::vector<double> t2 = {1.5, -2.0 / (1.0 + 2.0), 0.0};
  double alpha = 123.456;
  EXPECT_EQ(n.isrlu(t1), t2);
}

TEST(FunctionTesting, test_isrlu_large) {
  std::vector<double> t1 = {100.0, -50.0, 0.0};
  std::vector<double> t2 = {100.0, -50.0 / (1.0 + 0.5), 0.0};
  double alpha = 123.456;
  EXPECT_EQ(n.isrlu(t1), t2);
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
