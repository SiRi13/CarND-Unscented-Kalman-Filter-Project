#ifndef UKF_H
#define UKF_H

#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Dense"
#include "ground_truth_package.hpp"
#include "measurement_package.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
 private:
  void AugmentedSigmaPoints(MatrixXd* Xsig_aug);
  void SigmaPointPrediction(const MatrixXd Xsig_aug, double delta_t);
  void PredictMeanAndCovariance();
  void PredictRadarMeasurement(MatrixXd* Zsig, VectorXd* z_pred, MatrixXd* S);
  void UpdateState(const MeasurementPackage meas_package, const MatrixXd Zsig,
                   const VectorXd z_pred, const MatrixXd S);

 public:
  Tools tools_;

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* Sigma point spreading parameter
  double lambda_x_;
  double lambda_aug_;

  // radar measurement dimension
  int n_z_radar_;

  // lidar measurement dimension
  int n_z_laser_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  // amount of sigma points
  int n_sig_x_;

  // amount of augmented sigma points
  int n_sig_aug_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  double NIS_radar_;

  double NIS_lidar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  // process noise covariance matrix
  MatrixXd Q_;

  // measurement noise covariance matrix radar
  MatrixXd R_radar_;

  // measurement noise covariance matrix lidar
  MatrixXd R_lidar_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
