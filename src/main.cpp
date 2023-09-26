
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

double soft(double a) {
    return a / (1 + abs(a));
}

std::vector<double> softsign(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = soft(x[i]);
    return result;
}

TEST(FunctionTesting, testSoftSign1){
    EXPECT_NEAR(soft(-1.6),-0.615,1e-3);
    EXPECT_NEAR(soft(0.6),0.375,1e-3);
    EXPECT_NEAR(soft(0),0,1e-3);
}

TEST(FunctionTesting, testSoftSign2){
    EXPECT_NEAR(soft(0.15),0.13,1e-3);
    EXPECT_NEAR(soft(0.59),0.59,1e-3);
    EXPECT_NEAR(soft(-0.9),-0.474,1e-3);
}

TEST(FunctionTesting, testSoftSignPos){
    std::vector<double> x1 = {0.56, 0.99, 1.8, 2.1, 0.53};
    std::vector<double> right_x1 = {0.359, 0.497, 0.643, 0.677, 0.346};
    ASSERT_EQ(softsign(x1),right_x1);
}

TEST(FunctionTesting, testSoftSignMix){
    std::vector<double> x2 = {0.5, -0.4, -0.33, 0.1, -0.92};
    std::vector<double> right_x2 = {0.33, -0.286, -0.248, 0.09, -0.479};
    ASSERT_EQ(softsign(x2),right_x2);
}

TEST(FunctionTesting, testSoftSignNeg){
    std::vector<double> x3 = {-0.75, -0.93, -0.38, -0.02, -0.63};
    std::vector<double> right_x3 = {-0.429, -0.481, -0.275, -0.0196, -0.387};
    ASSERT_EQ(softsign(x3),right_x3);
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
