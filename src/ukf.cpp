#include "ukf.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // initialized?
  is_initialized_ = false;

  // initial state vector
  x_ = VectorXd::Ones(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  time_us_ = 0.0;

  NIS_radar_ = 0.0;
  NIS_lidar_ = 0.0;

  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_);

  weights_ = VectorXd::Constant(n_sig_, (0.5 * (lambda_aug_ + n_aug_)));
  weights_(0) = (lambda_aug_ / (lambda_aug_ + n_aug_));
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // skip if sensor_type_ should be ignored
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)) {
    return;
  }
  if (!is_initialized_) {
    // init x_, P_, time_us_
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      cout << "radar ";
      // convert radar from polar to cartesian coordinates
      x_(0) = meas_package.raw_measurements_(0) *
              cos(meas_package.raw_measurements_(1));
      x_(1) = meas_package.raw_measurements_(0) *
              sin(meas_package.raw_measurements_(1));

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      cout << "lidar ";
      // already cartesian coordinates
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    }

    // init previous timestamp
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    cout << "init done; x_=" << endl
         << x_ << endl
         << "P_ = " << endl
         << P_ << endl;
    // no need to predict or update
    return;
  }

  // calculate delta_t in seconds!
  float delta_t = (meas_package.timestamp_ - time_us_) / 1e+6;
  time_us_ = meas_package.timestamp_;
  cout << "delta_t done; start prediction" << endl;
  // predict
  Prediction(delta_t);

  // update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    cout << "prediction done; start udpate radar" << endl;
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    cout << "prediction done; start udpate lidar" << endl;
    UpdateLidar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;

  return;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  int i = 0;

  /*****************************************
   *          Create Sigma Points          *
   *****************************************/
  MatrixXd Xsig = MatrixXd(n_x_, n_sig_);
  MatrixXd A = P_.llt().matrixL();
  Xsig.col(0) = x_;
  A *= sqrt(lambda_x_ + n_x_);
  for (i = 0; i < n_x_; ++i) {
    Xsig.col(i + 1) = x_ + A.col(i);
    Xsig.col(i + n_x_ + 1) = x_ - A.col(i);
  }
  cout << "Xsig = " << endl << Xsig << endl;

  /*****************************************
   *     Create Augmented Sigma Points      *
   *****************************************/
  // create augmented mean state
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;
  // create augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;
  // create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();
  // create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_);
  A_aug *= sqrt(lambda_aug_ + n_aug_);
  for (i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + A_aug.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - A_aug.col(i);
  }
  cout << "Xsig_aug = " << endl << Xsig_aug << endl;

  /*****************************************
   *         Predict Sigma Points           *
   *****************************************/
  double delta_t_2 = (delta_t * delta_t) / 2;
  for (i = 0; i < n_sig_; ++i) {
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    Xsig_pred_.col(i) = Xsig_aug.block(0, i, n_x_, 1);
    if (fabs(yawd) > 0.001) {
      double curve_radius = v / yawd;
      double change_direction = yaw + (yawd * delta_t);
      Xsig_pred_(0, i) += curve_radius * (sin(change_direction) - sin(yaw));
      Xsig_pred_(1, i) += curve_radius * (cos(yaw) - cos(change_direction));
    } else {
      Xsig_pred_(0, i) += v * cos(yaw) * delta_t;
      Xsig_pred_(1, i) += v * sin(yaw) * delta_t;
    }
    Xsig_pred_(3, i) += yawd * delta_t;

    VectorXd nu_k = VectorXd::Zero(n_x_);
    nu_k << (delta_t_2 * cos(yaw) * nu_a), (delta_t_2 * sin(yaw) * nu_a),
        (delta_t * nu_a), (delta_t_2 * nu_yawdd), (delta_t * nu_yawdd);
    Xsig_pred_.col(i) += nu_k;
  }
  cout << "Xsig_pred_ = " << endl << Xsig_pred_ << endl;

  /*****************************************
   *     Predict Mean and Covariance        *
   *****************************************/
  // predict state mean
  for (i = 0; i < n_sig_; ++i) {
    x_ += (weights_(i) * Xsig_pred_.col(i));
  }

  // predict state covariance
  for (i = 0; i < n_sig_; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    P_ += (weights_(i) * x_diff * x_diff.transpose());
  }
  cout << "x_ / P_: " << endl << x_ << endl << P_ << endl;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int i = 0;
  MatrixXd Zsig = MatrixXd(n_z_laser_, n_sig_);
  VectorXd z_pred = VectorXd::Zero(n_z_laser_);
  MatrixXd S = MatrixXd::Zero(n_z_laser_, n_z_laser_);

  /*****************************************
   *     Predict Lidar Sigma Points        *
   *****************************************/
  // transform sigma points to measurement space
  // and
  // calculate mean predicted measurement
  for (i = 0; i < n_sig_; ++i) {
    // sigma points to measurement space
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
    // mean predicted measurement
    z_pred += (weights_(i) * Zsig.col(i));
  }
  // measurement covariance matrix S
  for (i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += (weights_(i) * z_diff * z_diff.transpose());
  }
  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z_laser_, n_z_laser_);
  R << (std_laspx_ * std_laspx_), 0, 0, (std_laspy_ * std_laspy_);
  S += R;

  /*****************************************
   *     Update State and Covariance       *
   *****************************************/
  // x-correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_laser_);
  for (i = 0; i < n_sig_; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc += (weights_(i) * x_diff * z_diff.transpose());
  }
  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();
  // measurement
  VectorXd z = meas_package.raw_measurements_;
  // update state mean and covariance
  VectorXd z_diff = z - z_pred;
  P_ -= (K * S * K.transpose());
  x_ += (K * z_diff);

  /*****************************************
   *            Calculate NIS              *
   *****************************************/
  NIS_lidar_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int i = 0;
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sig_);
  VectorXd z_pred = VectorXd::Zero(n_z_radar_);
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  /*****************************************
   *     Predict Radar Sigma Points        *
   *****************************************/
  // transform sigma points to measurement space
  // and
  // calculate mean predicted measurement
  for (i = 0; i < n_sig_; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double rho = sqrt((px * px) + (py * py));
    double phi = atan2(py, px);

    double rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v);
    if (fabs(rho) > 0.001) {
      rho_dot /= rho;
    } else {
      rho_dot = 0.0;
    }

    // sigma points to measurement space
    Zsig.col(i) << rho, phi, rho_dot;
    // mean predicted measurement
    z_pred += (weights_(i) * Zsig.col(i));
  }
  cout << "Xsig_pred_: " << endl << Xsig_pred_ << endl;
  cout << "Zsig / z_pred: " << endl << Zsig << endl << z_pred << endl;
  // calculate measurement covariance matrix S
  for (i = 0; i < n_sig_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    S += (weights_(i) * z_diff * z_diff.transpose());
  }
  cout << "S: " << endl << S << endl;
  // add process noise R to covariance matrix S
  MatrixXd R = MatrixXd(n_z_radar_, n_z_radar_);
  R << (std_radr_ * std_radr_), 0, 0, 0, (std_radphi_ * std_radphi_), 0, 0, 0,
      (std_radrd_ * std_radrd_);
  S += R;
  cout << "S + R: " << endl << S << endl;

  /*****************************************
   *     Update State and Covariance       *
   *****************************************/
  // calculate x-correlation matrix Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
  for (i = 0; i < n_sig_; ++i) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;

    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
    while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

    Tc += (weights_(i) * x_diff * z_diff.transpose());
  }
  // calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();
  cout << "Tc / K: " << endl << Tc << endl << K << endl;
  // actual measurement
  VectorXd z = meas_package.raw_measurements_;
  // udpate state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  while (z_diff(1) > M_PI) z_diff(1) -= 2.0 * M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.0 * M_PI;

  P_ -= (K * S * K.transpose());
  x_ += (K * z_diff);

  /*****************************************
   *            Calculate NIS              *
   *****************************************/
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
