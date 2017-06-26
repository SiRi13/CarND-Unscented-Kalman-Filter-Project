#include "tools.h"
#include <iostream>
#include "Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::Ref;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE here.
   */
  VectorXd rmse = VectorXd::Zero(4);

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Either Sizes don't match or size == 0" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned i = 0; i < estimations.size(); ++i) {
    VectorXd tmp = estimations[i] - ground_truth[i];
    tmp = tmp.array() * tmp.array();
    rmse += tmp;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

void Tools::NormalizeAngle(double &angle) {
  angle = atan2(sin(angle), cos(angle));
}
