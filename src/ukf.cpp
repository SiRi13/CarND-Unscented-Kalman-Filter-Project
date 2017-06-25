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

  // use lidar data
  use_laser_ = false;

  // use radar data
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

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

  // timestamp in microseconds
  time_us_ = 0.0;

  // NIS for radar
  NIS_radar_ = 0.0;

  // NIS for lidar
  NIS_lidar_ = 0.0;

  // lidar measurement dimension
  n_z_laser_ = 2;

  // radar measurement dimension
  n_z_radar_ = 3;

  // state dimension
  n_x_ = 5;

  // augmented state dimension
  n_aug_ = 7;

  // sigma points dimension
  n_sig_x_ = 2 * n_x_ + 1;

  // augmented sigma points dimension
  n_sig_aug_ = 2 * n_aug_ + 1;

  // spreading parameters
  lambda_x_ = 3 - n_x_;
  lambda_aug_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd::Ones(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // init predicted sigma points
  Xsig_pred_ = MatrixXd::Zero(n_x_, n_sig_aug_);

  Q_ = MatrixXd(2, 2);
  Q_ << (std_a_ * std_a_), 0, 0, (std_yawdd_ * std_yawdd_);

  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << (std_radr_ * std_radr_), 0, 0, 0, (std_radphi_ * std_radphi_), 0,
      0, 0, (std_radrd_ * std_radrd_);

  R_lidar_ = MatrixXd(n_z_laser_, n_z_laser_);
  R_lidar_ << (std_laspx_ * std_laspx_), 0, 0, (std_laspy_ * std_laspy_);

  weights_ = VectorXd::Constant(n_sig_aug_, (1 / (2 * (lambda_aug_ + n_aug_))));
  weights_(0) = (lambda_aug_ / (lambda_aug_ + n_aug_));
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
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

  // skip if sensor_type_ should be ignored
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)) {
    // calculate delta_t in seconds!
    float dt = (meas_package.timestamp_ - time_us_) / 1e+6;
    time_us_ = meas_package.timestamp_;

    // cout << "dt done; start prediction" << endl;

    // predict
    Prediction(dt);

    // update
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // cout << "prediction done; start udpate radar" << endl;
      UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // cout << "prediction done; start udpate lidar" << endl;
      UpdateLidar(meas_package);
    }

    // print the output
    // cout << "x_ = " << x_ << endl;
    // cout << "P_ = " << P_ << endl;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_aug_);
  AugmentedSigmaPoints(&Xsig_aug);
  SigmaPointPrediction(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int i = 0;
  MatrixXd Zsig = MatrixXd(n_z_laser_, n_sig_aug_);
  VectorXd z_pred = VectorXd::Zero(n_z_laser_);
  MatrixXd S = MatrixXd::Zero(n_z_laser_, n_z_laser_);

  /*****************************************
   *     Predict Lidar Sigma Points        *
   *****************************************/
  // transform sigma points to measurement space
  // and
  // calculate mean predicted measurement
  for (i = 0; i < n_sig_aug_; ++i) {
    // sigma points to measurement space
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
    // mean predicted measurement
    z_pred += (weights_(i) * Zsig.col(i));
  }
  // measurement covariance matrix S
  for (i = 0; i < n_sig_aug_; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += (weights_(i) * z_diff * z_diff.transpose());
  }
  // add measurement noise covariance matrix
  S += R_lidar_;

  /*****************************************
   *     Update State and Covariance       *
   *****************************************/
  // x-correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_laser_);
  for (i = 0; i < n_sig_aug_; ++i) {
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
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sig_aug_);
  VectorXd z_pred = VectorXd::Zero(n_z_radar_);
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  PredictRadarMeasurement(&Zsig, &z_pred, &S);
  UpdateState(meas_package, Zsig, z_pred, S);
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out) {
  // Augmented Mean State
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  // aug covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  // Augmented Covariance Matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner<2, 2>() << Q_;

  // square root of P_aug
  MatrixXd A_aug = P_aug.llt().matrixL();

  // sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, n_sig_aug_);
  // set values in output matrix
  Xsig_aug.col(0) = x_aug;
  // calculate sigma points ...
  A_aug *= sqrt(lambda_aug_ + n_aug_);
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1) = x_aug + A_aug.col(i);
    Xsig_aug.col(i + n_aug_ + 1) = x_aug - A_aug.col(i);
  }

  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd Xsig_aug, double delta_t) {
  MatrixXd Xsig_pred = MatrixXd(n_x_, n_sig_aug_);

  // predict sigma points
  for (int i = 0; i < n_sig_aug_; ++i) {
    MatrixXd sig_point = Xsig_aug.col(i);
    Xsig_pred.col(i) = sig_point.topRows(n_x_);

    if (sig_point(4) > 0) {
      double _2over4 = (sig_point(2) / sig_point(4));
      double _3plus4timesDelta = sig_point(3) + sig_point(4) * delta_t;
      Xsig_pred(0, i) +=
          (_2over4 * (sin(_3plus4timesDelta) - sin(sig_point(3))));
      Xsig_pred(1, i) +=
          (_2over4 * (cos(sig_point(3)) - cos(_3plus4timesDelta)));
      Xsig_pred(3, i) += (sig_point(4) * delta_t);
    } else {
      Xsig_pred(0, i) += (sig_point(2) * cos(sig_point(3)) * delta_t);
      Xsig_pred(1, i) += (sig_point(2) * sin(sig_point(3)) * delta_t);
      Xsig_pred(3, i) += (sig_point(4) * delta_t);
    }

    double delta_t_2 = 0.5 * (delta_t * delta_t);
    VectorXd nu_k = VectorXd::Zero(n_x_);
    nu_k << (delta_t_2 * cos(sig_point(3)) * sig_point(5)),
        (delta_t_2 * sin(sig_point(3)) * sig_point(5)),
        (delta_t * sig_point(5)), (delta_t_2 * sig_point(6)),
        (delta_t * sig_point(6));
    Xsig_pred.col(i) += nu_k;
  }

  // write result
  Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance() {
  VectorXd x = VectorXd::Zero(n_x_);
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);

  // predict state mean
  for (int i = 0; i < n_sig_aug_; ++i) {
    x += (weights_(i) * Xsig_pred_.col(i));
  }

  // predict state covariance
  for (int i = 0; i < n_sig_aug_; ++i) {
    // P += (weights(i) * (Xsig_pred.col(i) - x) * (Xsig_pred.col(i) -
    // x).transpose());
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    // angle normalization
    x_diff(3) = tools_.NormalizeAngle(x_diff(3));
    P += weights_(i) * x_diff * x_diff.transpose();
  }

  // print results
  // cout << "Predicted state: " << endl << x << endl;
  // cout << "Predicted Covariance matrix: " << endl << P << endl;

  // write result
  x_ = x;
  P_ = P;
}

void UKF::PredictRadarMeasurement(MatrixXd* Zsig_out, VectorXd* z_out,
                                  MatrixXd* S_out) {
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, n_sig_aug_);
  VectorXd z_pred = VectorXd::Zero(n_z_radar_);
  MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);

  for (int i = 0; i < n_sig_aug_; ++i) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double rho = sqrt((px * px) + (py * py));
    double phi = atan2(py, px);
    double rho_dot = (px * cos(yaw) * v + py * sin(yaw) * v) / rho;

    // transform sigma points into measurement space
    Zsig.col(i) << rho, phi, rho_dot;
    // calculate mean predicted measurement
    z_pred += (weights_(i) * Zsig.col(i));
  }

  // calculate measurement covariance matrix S
  for (int i = 0; i < n_sig_aug_; ++i) {
    MatrixXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    z_diff(1) = tools_.NormalizeAngle(z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_radar_;

  // print results
  // cout << "z_pred = " << endl << z_pred << endl;
  // cout << "S = " << endl << S << endl;

  // write results
  *Zsig_out = Zsig;
  *z_out = z_pred;
  *S_out = S;
}

void UKF::UpdateState(const MeasurementPackage meas_package,
                      const MatrixXd Zsig, const VectorXd z_pred,
                      const MatrixXd S) {
  // create example vector for incoming radar measurement
  VectorXd z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);

  // calculate cross correlation matrix
  int i = 0;
  for (i = 0; i < n_sig_aug_; ++i) {
    MatrixXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools_.NormalizeAngle(x_diff(3));

    MatrixXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = tools_.NormalizeAngle(z_diff(1));

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }
  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // update state mean and covariance matrix
  P_ -= (K * S * K.transpose());
  x_ += (K * (z - z_pred));

  // print result
  // cout << "Updated state x: " << endl << x_ << endl;
  // cout << "Updated state covariance P: " << endl << P_ << endl;
}
