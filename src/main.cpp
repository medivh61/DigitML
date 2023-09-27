
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

double isrlu(double a, double alpha = 1.0) {
    return a >= 0 ? a : a / sqrt(1 + alpha * a * a);
}

std::vector<double> isrlusign(const std::vector<double>& x, double alpha = 1.0) {
    std::vector<double> result(x.size());
    for (unsigned int i = 0; i < x.size(); i++)
        result[i] = isrlu(x[i], alpha);
    return result;
}

TEST(FunctionTesting, testIsrluNegative) {
    double alpha = 0.5;
    EXPECT_NEAR(isrlu(-2.5, alpha), -1.4744195615489712, 1e-9);
    EXPECT_NEAR(isrlu(-0.7, alpha), -0.6536064526481745, 1e-9);
    EXPECT_NEAR(isrlu(-0.9, alpha), -0.759284173097276, 1e-9);
}

TEST(FunctionTesting, testIsrluPositive) {
    double alpha = 0.5;
    EXPECT_NEAR(isrlu(1.8, alpha), 1.8, 1e-9);
    EXPECT_NEAR(isrlu(0.2, alpha), 0.2, 1e-9);
    EXPECT_NEAR(isrlu(0.56, alpha), 0.56, 1e-9);
}

TEST(FunctionTesting, testIsrluZero) {
    double alpha = 0.5;
    EXPECT_NEAR(isrlu(0, alpha), 0, 1e-9);
    EXPECT_NEAR(isrlu(0.1, alpha), 0.1, 1e-9);
    EXPECT_NEAR(isrlu(0.0, alpha), 0.0, 1e-9);
}

//TEST(FunctionTesting, testIsrluMixed) {
  //  double alpha = 0.5;
    //std::vector<double> x = {0.5, -0.4, -0.33, 0.1, -0.92};
    //std::vector<double> result = isrlusign(x, alpha);
    //std::vector<double> expected = {0.5, -0.253568, -0.205501, 0.1, -0.393458};
    //for (unsigned int i = 0; i < result.size(); i++)
      //  EXPECT_NEAR(result[i], expected[i], 1e-9);
//}

#endif  // TESTS

int main(int argc, char **argv) {
   tests(6);
   #ifdef TESTS
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    #endif
     
        return 0;
}
